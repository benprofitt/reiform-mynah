from impl.services.modules.core.resources import *
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICDataSet

def make_predictions(embeddings, names, clf):

    predictions = clf.fit_predict(embeddings)

    cluster_ids = set(predictions)
    count = len(cluster_ids)

    clusters = []

    if count > 1:
        clusters = []
        for i in range(count):
            clusters.append([])
        min_ = min(cluster_ids)
            
        for i, id in enumerate(predictions):
            clusters[id - min_].append(names[i])

    return count, clusters

def perform_split(dataset : ReiformICDataSet, 
                  projection_label : str, clf_generator : Callable, 
                  classes : List[str]) -> Tuple[Dict[str, int], Dict[str, List[List[str]]]]:

    potential_cluster_count : Dict[str, int] = {}        # class -> number of new classes
    potential_clusters : Dict[str, List[List[str]]] = {} # class -> List of "New_Classes = (Lists of file uuids)"

    for c in classes:

        count, clusters = cluster_class(dataset, projection_label, clf_generator, c)

        potential_cluster_count[c] = count
        potential_clusters[c] = clusters

    return potential_cluster_count, potential_clusters

def cluster_class(dataset, projection_label, clf_generator, c):
    embeddings : List[NDArray] = []
    names : List[str] = []

    for name, file in dataset.get_items(c):
        embeddings.append(file.get_projection(projection_label)) 
        names.append(name)

    clf = clf_generator() 
    count, clusters = make_predictions(embeddings, names, clf)
    return count, clusters
