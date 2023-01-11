from resources import *

def find_images_near_gaps(dataset : ReiformICDataSet, cls: str) -> List[List[List[Any]]]:

    projection_label = GAP_PROJECTION_LABEL

    embeddings : List[NDArray] = []
    names : List[str] = []

    for name, file in dataset.get_items(cls):
        embeddings.append(file.get_projection(projection_label)) 
        names.append(name)

    clf = OPTICS(min_samples=len(embeddings)//15)
    predictions = clf.fit_predict(embeddings)
    cluster_classes = set(predictions)
    if len(cluster_classes) < 2:
        return [[[]]]
    clusters = [[] for _ in cluster_classes]
    cluster_points = [[] for _ in cluster_classes]

    for pred, name, point in zip(predictions, names, embeddings):
        clusters[pred].append(dataset.get_file(cls, name))
        cluster_points[pred].append(point)

    cluster_pair_results = []
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters[i+1:]):

            cluster_pair_results.append(find_points_near_border((c1, c2), (cluster_points[i], cluster_points[i+1+j])))

    return cluster_pair_results

def find_images_around_holes():
    pass