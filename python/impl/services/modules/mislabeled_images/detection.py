from sklearn import svm # type: ignore
# from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier # type: ignore
# from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier # type: ignore
from impl.services.modules.mislabeled_images.mislabeled_resources import *
import cupy

def estimate_outlier_ratio(projected_data : List[NDArray]) -> float:
    
    arr = np.array(projected_data)

    mean = np.mean(arr, 0)
    std_dev = np.std(arr, 0)

    sigma_count = 1.5

    plus_three_sigma = mean + sigma_count * std_dev
    minus_three_sigma = mean - sigma_count * std_dev

    outlier_count = 0

    for v in projected_data:
        if np.any(v>plus_three_sigma) or np.any(v<minus_three_sigma):
            outlier_count += 1

    ReiformInfo("Outlier Ratio Used: {}".format(round(outlier_count/len(projected_data), 3)))

    return outlier_count/len(projected_data)

def find_outliers_loda(projected_data : ReiformICDataSet, label : str, cuts : int) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    
    inlier_results : ReiformICDataSet = ReiformICDataSet(projected_data.classes())
    outlier_results: ReiformICDataSet = ReiformICDataSet(projected_data.classes())

    detection_results = (inlier_results, outlier_results)
    
    for class_key in projected_data.classes():

        X : List[NDArray] = []
        files : List[ReiformICFile] = []

        for _, file in projected_data.get_items(class_key):
            X.append(file.get_projection(label))
            files.append(file)

        outlier_ratio : float = max(estimate_outlier_ratio(X), 0.005)
        
        detector = LODA(None, cuts)

        start = time.time()
        X = cupy.array(np.array(X))
        detector.fit(X)
        scores = detector.score(X)
        
        ReiformInfo("Detection fit: {}".format(round(time.time()-start, 3)))

        order = cupy.argsort(scores)

        for pos, file in zip(order, files):
            if pos > len(files) * (1-(outlier_ratio)):
                detection_results[1].add_file(file)
            else:
                detection_results[0].add_file(file)


    return detection_results

def predict_outlier_labels(inliers : ReiformICDataSet, 
                           outliers : ReiformICDataSet) -> Tuple[ReiformICDataSet, 
                                                                 ReiformICDataSet]:
    # Quick "guess"
    clf = cuml.neighbors.KNeighborsClassifier(n_neighbors=max(5, min(150, inliers.file_count()//100)))

    X_known = []
    y_known = []

    X_unknown = []
    y_unknown = []
    name_unknown = []

    for i, c in enumerate(inliers.classes()):
        for _, file in inliers.get_items(c):
            X_known.append(file.get_projection(PROJECTION_LABEL_REDUCED_EMBEDDING))
            y_known.append(i)

    clf.fit(np.array(X_known), np.array(y_known))
    for i, c in enumerate(outliers.classes()):
        for name, file in outliers.get_items(c):
            X_unknown.append(file.get_projection(PROJECTION_LABEL_REDUCED_EMBEDDING))
            y_unknown.append(c)
            name_unknown.append(name)

    preds = clf.predict(np.array(X_unknown))

    prob_preds = clf.predict_proba(np.array(X_unknown))
    
    for name, p_class, c_probs, o_class in zip(name_unknown, preds, prob_preds, y_unknown):
        
        file = outliers.get_file(o_class, name)
        if inliers.classes()[p_class] == o_class and c_probs[p_class] > 0.75:
            outliers.remove_file(o_class, name)
            inliers.add_file(file)
        else:
            file.add_projection(NEW_LABEL_PREDICTION, np.array([p_class]))
            file.add_projection(NEW_LABEL_PREDICTION_PROBABILITIES, c_probs)

    return inliers, outliers

def find_outlier_consensus(dataset : ReiformICDataSet):
    
    projections : List[str] = [
                               PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS
                              ]

    possible_detectors : List[Callable] = [
                                           find_outliers_loda
                                          ]
    possible_cuts : List[int] = [1251, 1103, 1009, 931, 871, 767, 1133, 1039, 921, 831, 737, 1143, 1549, 951, 541, 567]

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
    
    if outlier_results.empty():
        ReiformInfo("No Outliers")
        return inlier_results, outlier_results
    else:
        datasets = predict_outlier_labels(inlier_results, outlier_results)

    return datasets