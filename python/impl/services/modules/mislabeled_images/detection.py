from impl.services.modules.mislabeled_images.mislabeled_resources import *

def estimate_outlier_ratio(projected_data : ReiformICDataSet) -> float:
    #TODO: Research how to predict/estimate this type of value
    return 0.1

def find_outliers_loda(projected_data : ReiformICDataSet, outlier_ratio : float, label : str) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    detector = LODA(outlier_ratio, "auto", 1000)
    
    inlier_results : ReiformICDataSet = ReiformICDataSet(projected_data.classes())
    outlier_results: ReiformICDataSet = ReiformICDataSet(projected_data.classes())

    detection_results = (inlier_results, outlier_results)
    
    for class_key in projected_data.classes():
        X = []
        files = []

        for _, file in projected_data.get_items(class_key):
            X.append(file.get_projection(label))
            files.append(file)

        detector.fit(X)
        outlier_labels = detector.labels_

        for file, l in zip(files, outlier_labels):
            detection_results[l].add_file(file)

    return detection_results


def find_outliers_isolation_forest(projected_data : ReiformICDataSet, outlier_ratio : float, label : str):

    detector = IsolationForest(contamination=outlier_ratio)

    return find_outliers_with_detector(projected_data, label, detector)


def find_outliers_with_detector(projected_data : ReiformICDataSet, label : str, detector):
    inlier_results : ReiformICDataSet = ReiformICDataSet(projected_data.classes())
    outlier_results: ReiformICDataSet = ReiformICDataSet(projected_data.classes())

    detection_results = (outlier_results, inlier_results)
    
    for class_key in projected_data.classes():
        X = []
        files = []

        for _, file in projected_data.get_items(class_key):
            X.append(file.get_projection(label))
            files.append(file)

        outlier_labels = detector.fit_predict(X)
        
        outlier_labels[outlier_labels == -1] = 0

        for file, inlier_label in zip(files, outlier_labels):
            detection_results[inlier_label].add_file(file)

    detection_results = (detection_results[1], detection_results[0])
    return detection_results

def find_outlier_consensus(dataset : ReiformICDataSet):
    ratio : float = estimate_outlier_ratio(dataset)
    projections : List[str] = [
                               PROJECTION_LABEL_REDUCED_EMBEDDING,
                               PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS
                              ]

    possible_detectors : List[Callable] = [
                                           find_outliers_loda,
                                           find_outliers_isolation_forest
                                          ]
    
    # Keeps track of the "votes" for various files to be inliers vs outliers.... 
    # 2/3 majority for out? We'll try it!
    votes : Dict[str, Dict[str, int]] = {}
    total_possible : int = 0
    for c in dataset.classes():
        votes[c] = {}

    for i, proj in enumerate(projections):
        # This is used to increase the weight of the "better" embeddings. Still gives a good sample
        weight = 5 - i
        for det in possible_detectors:
            total_possible += weight
            _, outliers = det(dataset, ratio, proj)
                                
            for c in outliers.classes():
                for name, _ in outliers.get_items(c):
                    if name not in votes[c]:
                        votes[c][name] = 0
                    votes[c][name] += weight
    
    inlier_results : ReiformICDataSet = ReiformICDataSet(dataset.classes())
    outlier_results: ReiformICDataSet = ReiformICDataSet(dataset.classes())

    for c, names in votes.items():
        for name, count in names.items():
            if count > (total_possible * (9/20)):
                outlier_results.add_file(dataset.get_file(c, name))
            else:
                inlier_results.add_file(dataset.get_file(c, name))

        for filename, file in dataset.get_items(c):
            if filename not in names:
                inlier_results.add_file(file)
    
    return inlier_results, outlier_results