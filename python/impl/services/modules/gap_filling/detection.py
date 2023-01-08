from resources import *

def find_images_near_gaps(dataset : ReiformICDataSet, cls: str):

    projection_label = PROJECTION_LABEL_2D_PER_CLASS

    embeddings : List[NDArray] = []
    names : List[str] = []

    for name, file in dataset.get_items(cls):
        embeddings.append(file.get_projection(projection_label)) 
        names.append(name)

    clf = OPTICS(min_samples=len(embeddings)//5)
    predictions = clf.fit_predict(embeddings)
    clusters = [[] for _ in set(predictions)]

    for pred, name in zip(predictions, names):
        clusters[pred].append(dataset.get_file(cls, name))

    cluster_pair_results = []
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:

            cluster_pair_results.append(find_points_near_border((c1, c2)))

    return cluster_pair_results
