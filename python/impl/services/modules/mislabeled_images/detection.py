from impl.services.modules.mislabeled_images.mislabeled_resources import *

# shape of latent projection:
# {
#    0: {
#           "image_name" : <np.array> len = latent dimensions
#       }
#    1: ...
# ...
# }

def estimate_outlier_ratio(projected_data : ReiformICDataSet) -> float:
    pass

def find_outliers_loda(projected_data : ReiformICDataSet, outlier_ratio : float) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    detector = LODA(outlier_ratio, "auto", 1000)
    
    inlier_results : ReiformICDataSet = ReiformICDataSet(projected_data.classes())
    outlier_results: ReiformICDataSet = ReiformICDataSet(projected_data.classes())

    detection_results = (inlier_results, outlier_results)
    
    for class_key in projected_data.classes():
        X = []
        files = []

        for _, file in projected_data.get_items(class_key):
            X.append(file.get_projection(PROJECTION_LABEL))
            files.append(file)

        detector.fit(X)
        outlier_labels = detector.labels_

        for file, label in zip(files, outlier_labels):
            detection_results[label].add_file(file)

    return detection_results


def find_outliers_isolation_forest(projected_data : ReiformICDataSet, outlier_ratio : float):


    detector = IsolationForest(contamination=outlier_ratio)

    inlier_results : ReiformICDataSet = ReiformICDataSet(projected_data.classes())
    outlier_results: ReiformICDataSet = ReiformICDataSet(projected_data.classes())

    detection_results = (inlier_results, outlier_results)
    
    for class_key in projected_data.classes():
        X = []
        files = []

        for _, file in projected_data.get_items(class_key):
            X.append(file.get_projection(PROJECTION_LABEL))
            files.append(file)

        outlier_labels = detector.fit_predict(X)
        
        outlier_labels[outlier_labels == -1] = 0

        for file, label in zip(files, outlier_labels):
            
            detection_results[label].add_file(file)

    return detection_results