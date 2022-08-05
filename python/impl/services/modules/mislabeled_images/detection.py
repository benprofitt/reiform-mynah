from sklearn import svm # type: ignore
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier # type: ignore
from impl.services.modules.mislabeled_images.mislabeled_resources import *

def estimate_outlier_ratio(projected_data : List[NDArray]) -> float:
    
    arr = np.array(projected_data)

    mean = np.mean(arr, 0)
    std_dev = np.std(arr, 0)

    sigma_count = 2.5

    plus_three_sigma = mean + sigma_count * std_dev
    minus_three_sigma = mean - sigma_count * std_dev

    outlier_count = 0

    for v in projected_data:
        if np.any(v>plus_three_sigma) or np.any(v<minus_three_sigma):
            outlier_count += 1

    # ReiformInfo("Outlier Ratio Used: {}".format(round(outlier_count/len(projected_data), 3)))

    return outlier_count/len(projected_data)

def find_outliers_loda(projected_data : ReiformICDataSet, label : str, cuts : int) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    
    inlier_results : ReiformICDataSet = ReiformICDataSet(projected_data.classes())
    outlier_results: ReiformICDataSet = ReiformICDataSet(projected_data.classes())

    detection_results = (inlier_results, outlier_results)
    
    for class_key in projected_data.classes():

        # ReiformInfo("Starting detection round for {} with {}".format(class_key, label))

        X : List[NDArray] = []
        files : List[ReiformICFile] = []

        for _, file in projected_data.get_items(class_key):
            X.append(file.get_projection(label))
            files.append(file)

        outlier_ratio : float = estimate_outlier_ratio(X)
        detector = LODA(outlier_ratio, "auto", cuts)

        detector.fit(X)
        outlier_labels = detector.labels_

        for file, l in zip(files, outlier_labels):
            detection_results[l].add_file(file)

    return detection_results

def predict_outlier_labels(inliers : ReiformICDataSet, 
                           outliers : ReiformICDataSet) -> Tuple[ReiformICDataSet, 
                                                                 ReiformICDataSet]:
    # Quick "guess"
    clf = KNeighborsClassifier(n_neighbors=min(150, inliers.file_count()//100))

    X_known = []
    y_known = []

    X_unknown = []
    y_unknown = []
    name_unknown = []

    for i, c in enumerate(inliers.classes()):
        for _, file in inliers.get_items(c):
            X_known.append(file.get_projection(PROJECTION_LABEL_REDUCED_EMBEDDING))
            y_known.append(i)

    clf.fit(X_known, y_known)

    for i, c in enumerate(outliers.classes()):
        for name, file in outliers.get_items(c):
            X_unknown.append(file.get_projection(PROJECTION_LABEL_REDUCED_EMBEDDING))
            y_unknown.append(c)
            name_unknown.append(name)

    preds = clf.predict(X_unknown)
    prob_preds = clf.predict_proba(X_unknown)
    
    for name, p_class, c_probs, o_class in zip(name_unknown, preds, prob_preds, y_unknown):
        
        outliers.get_file(o_class, name).add_projection(NEW_LABEL_PREDICTION, np.array([p_class]))
        outliers.get_file(o_class, name).add_projection(NEW_LABEL_PREDICTION_PROBABILITIES, c_probs)

    return inliers, outliers

def find_outlier_consensus(dataset : ReiformICDataSet):
    
    projections : List[str] = [
                               PROJECTION_LABEL_REDUCED_EMBEDDING,
                               PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS
                              ]

    possible_detectors : List[Callable] = [
                                           find_outliers_loda
                                          ]
    possible_cuts : List[int] = [1103, 1009, 931, 871, 767]

    # Keeps track of the "votes" for various files to be inliers vs outliers.... 
    votes : Dict[str, Dict[str, int]] = {}
    total_possible : int = 0
    for c in dataset.classes():
        votes[c] = {}

    for i, proj in enumerate(projections):
        # This is used to increase the weight of the "better" embeddings. Still gives a good sample
        weight = 5 - i
        for det in possible_detectors:
            for cuts in possible_cuts:
                total_possible += weight
                _, outliers = det(dataset, proj, cuts)
                                    
                for c in outliers.classes():
                    for name, _ in outliers.get_items(c):
                        if name not in votes[c]:
                            votes[c][name] = 0
                        votes[c][name] += weight
    
    inlier_results : ReiformICDataSet = ReiformICDataSet(dataset.classes())
    outlier_results: ReiformICDataSet = ReiformICDataSet(dataset.classes())

    for c, names in votes.items():
        for name, count in names.items():
            if count > (total_possible * (1/2)):
                outlier_results.add_file(dataset.get_file(c, name))
            else:
                inlier_results.add_file(dataset.get_file(c, name))

        for filename, file in dataset.get_items(c):
            if filename not in names:
                inlier_results.add_file(file)
    
    return predict_outlier_labels(inlier_results, outlier_results)