from impl.services.modules.mislabeled_images.mislabeled_resources import *

def train_correction_model(dataset: ReiformICDataSet, edge_size: int) -> nn.Module:

    classes = len(dataset.classes())

    model = SmallAutoNet(insize, edge_size, classes)
    dataloader = dataset.get_dataloader(insize, edge_size, CORRECTION_MODEL_BATCH_SIZE)
    model, _ = train_conv_net(model, dataloader, multiclass_model_loss, 
                    get_optimizer(model), MONTE_CARLO_TRAINING_EPOCHS, MONTE_CARLO_TRAINING_EPOCHS//2)

    return model

def train_correction_model_ensemble(dataset : ReiformICDataSet) -> Tuple[List[nn.Module], List[int]]:
    
    models : List[nn.Module] = []
    pwr_2 = max(32, closest_power_of_2(dataset.max_size()))
    edgesizes = [pwr_2, pwr_2]
    
    for edge_size in edgesizes:
        model = train_correction_model(dataset, edge_size)
        models.append(model)

    return models, edgesizes

def predict_correct_label(dataset : ReiformICDataSet, model: nn.Module, edge_size: int) -> ReiformICDataSet:
    
    for c in dataset.classes():
        class_dataset : ReiformICDataSet = dataset.filter_classes(c)

        if class_dataset.file_count() == 0:
            continue

        dataloader = class_dataset.get_dataloader(insize, edge_size, 1)
        labeling_results : List[Tuple[str, NDArray]] = predict_labels(model, dataloader)

        for name, new_label in labeling_results:
            dataset.get_file(c, name).add_confidence_vector(new_label)

    return dataset

def predict_correct_label_ensemble(dataset : ReiformICDataSet, models: List[nn.Module], edgesizes: List[int]) -> ReiformICDataSet:

    for edge_size, model in zip(edgesizes, models):
        dataset = predict_correct_label(dataset, model, edge_size)

    return dataset

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

    if max_freq[max_max] >= total * (9/10):
        return True, max_max

    return False, -1
    

def monte_carlo_label_correction(simulations: int, 
                                 inliers: ReiformICDataSet, 
                                 outliers: ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    
    to_correct : ReiformICDataSet = outliers.copy()

    for i in range(simulations):
        ReiformInfo("Simulation: {}".format(i+1))
        incl, dinc = inliers.split(0.9)

        if outliers.file_count() == 0:
            to_correct = dinc

        # Use incl to train several types of ML/DL models
        models : List[nn.Module] = [] 
        edgesizes : List[int] = [] 

        models, edgesizes = train_correction_model_ensemble(incl)

        # Classify contents of 'to_correct' using new models
        to_correct = predict_correct_label_ensemble(to_correct, models, edgesizes)

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
        for key, data in to_correct.get_items(c):
            predictions = np.array(data.get_confidence_vectors())
            keep, new_label = evaluate_correction_confidence(i, predictions)
            data.clear_confidence_vectors()
            data.set_class(to_correct.classes()[new_label])
            if keep:
                corrected.add_file(data)
            else:
                dropped.add_file(data)

    return corrected, dropped
    

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