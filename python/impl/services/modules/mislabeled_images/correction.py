from sklearn import svm # type: ignore
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from impl.services.modules.mislabeled_images.mislabeled_resources import *

def train_classic_model(dataset : ReiformICDataSet, label : str, classifier : Any):

    X_known = []
    y_known = []

    for i, c in enumerate(dataset.classes()):
        for _, file in dataset.get_items(c):
            y_known.append(i)
            X_known.append(file.get_projection(label))

    classifier.fit(X_known, y_known)

    return classifier


def train_correction_model_ensemble(dataset : ReiformICDataSet) -> Tuple[List[Any], List[str]]:

    # Train one model for each of RF, GB, KNN, Ada, SVM(later)
    # Crossed with one for each embedding (at least reduced and 2D (maybe some combination), and maybe make more embeddings?)

    models : List[Any] = []
    labels : List[str] = []
    for embedding_label in [PROJECTION_LABEL_REDUCED_EMBEDDING]:
        models_to_train : List[Any] = [
            AdaBoostClassifier(n_estimators=661),
            KNeighborsClassifier(n_neighbors=min(100, dataset.file_count()//100), weights='distance'),
            GradientBoostingClassifier(n_estimators=421, max_depth=4, min_samples_leaf=4),
            RandomForestClassifier(n_estimators=450, min_samples_leaf=4)
        ]
        
        for model_num, clf in enumerate(models_to_train):
            start = time.time()
            labels.append(embedding_label)
            model = train_classic_model(dataset, embedding_label, clf)
            models.append(model)

    return models, labels


def predict_correct_label(outliers : ReiformICDataSet, classifier : Any, label : str):

    X_unknown = []
    y_unknown = []
    name_unknown = []

    for i, c in enumerate(outliers.classes()):
        for name, file in outliers.get_items(c):
            X_unknown.append(file.get_projection(label))
            y_unknown.append(c)
            name_unknown.append(name)

    preds = classifier.predict_proba(X_unknown)

    return name_unknown, y_unknown, preds

def predict_correct_label_ensemble(dataset : ReiformICDataSet, models: List[Any], labels: List[str]) -> List[Tuple[List[str], List[str], List[NDArray]]]:

    results : List[Tuple[List[str], List[str], List[NDArray]]] = []

    for label, model in zip(labels, models):
        results.append(predict_correct_label(dataset, model, label))

    return results

def evaluate_correction_confidence(starting_label: int, predictions: NDArray) -> Tuple[bool, int]:
    # This will be complex: We will evaluate the combination of all label confidences
    # from all models in the ensemble. Think about best way to do this.
    # Possible outcomes: remove, correct, remain

    # predictions => (P, C) ; P is number of predictions, C is number of classes

    # Compute the freq that something is the max in the array
    average = np.mean(predictions, axis=0)
    max_freq = [0] * average.size
    max_indices = np.argmax(predictions, axis=1)
    total : int = 0
    for val in max_indices:
        total += 1
        max_freq[int(val)] = max_freq[int(val)] + 1
    max_max = int(max_freq.index(max(max_freq)))

    if max_freq[max_max] >= total * (100/100):
        return True, max_max

    return False, -1
    
def monte_carlo_label_correction(simulations: int, 
                                 inliers: ReiformICDataSet, 
                                 outliers: ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    
    to_correct : ReiformICDataSet = outliers.copy()

    packages : List[Tuple[ReiformICDataSet, ReiformICDataSet]] = []

    for i in range(simulations):
        ReiformInfo("Simulation: {}".format(i+1))
        incl, dinc = inliers.split(0.9)
        if outliers.file_count() == 0:
            to_correct = dinc

        packages.append((incl, to_correct))

    with Pool(16) as p:
        results = p.map(monte_carlo_parallel, packages)

    for _, result_list in results:
        for result in result_list:
            name_unknown, y_unknown, preds = result
            for name, o_class, pred_vec in zip(name_unknown, y_unknown, preds):
                to_correct.get_file(o_class, name).add_confidence_vector(pred_vec)

    corrected : ReiformICDataSet = ReiformICDataSet(classes=outliers.classes())
    dropped : ReiformICDataSet = ReiformICDataSet(classes=outliers.classes())

    if outliers.empty():
        to_correct = inliers
        for i, c in enumerate(to_correct.classes()):
            for _, file in to_correct.get_items(c):
                if not file.has_confidence_vectors():
                    fake_pred = [0.0] * len(to_correct.classes())
                    fake_pred[i] = 1.0
                    file.add_confidence_vector(np.array(fake_pred))


    for i, c in enumerate(to_correct.classes()):
        for _, data in to_correct.get_items(c):
            predictions = np.array(data.get_confidence_vectors())
            keep, new_label = evaluate_correction_confidence(i, predictions)
            data.clear_confidence_vectors()
            data.set_class(to_correct.classes()[new_label])
            if keep:
                corrected.add_file(data)
            else:
                dropped.add_file(data)

    return corrected, dropped

def monte_carlo_parallel(package):

    # Get datasets out of the package
    incl, to_correct = package

    # Use incl to train several types of ML/DL models
    models : List[Any] = [] 
    labels : List[str] = [] 

    # New plan : Train models that are KNN, AdaBoost, SVM, RandomForest and GradientBoost
    models, labels = train_correction_model_ensemble(incl)

    # Classify contents of 'to_correct' using new models
    results = predict_correct_label_ensemble(to_correct, models, labels)
    return to_correct, results
    

def iterative_reinjection_label_correction(iterations : int, 
                                 inliers: ReiformICDataSet, 
                                 outliers: ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet, ReiformICDataSet]:
    start = time.time()
    all_corrected : ReiformICDataSet = ReiformICDataSet(inliers.classes())
    ReiformInfo("Inliers / Outliers counts: {} / {} ".format(inliers.file_count(), outliers.file_count()))
    for iter in range(iterations):
        ReiformInfo("Iteration: {} / {}".format(iter+1, iterations))

        corrected, dropped = monte_carlo_label_correction(MONTE_CARLO_SIMULATIONS, inliers, outliers)

        all_corrected.merge_in(corrected)

        if outliers.file_count() == 0:
            inliers.set_minus(corrected)
            inliers.set_minus(dropped)

        inliers.merge_in(corrected)

        if outliers == dropped:
            ReiformInfo("Early termination of iterative label correction: outliers == dropped (no improvement)")
            break

        outliers = dropped
        if outliers.file_count() == 0:
            ReiformInfo("Early termination of iterative label correction: |outliers| == 0 (all corrected)")
            break

        ReiformInfo("Inliers / Outliers / Dropped / Corrected / All_Corr counts: {} / {} / {} / {} / {}".format(
                            inliers.file_count(), outliers.file_count(), dropped.file_count(), 
                            corrected.file_count(), all_corrected.file_count()))

    ReiformInfo("Full Monte Carlo timing: {} seconds".format(round(time.time() - start) ))
    return inliers, outliers, all_corrected