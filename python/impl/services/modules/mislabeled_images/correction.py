from impl.services.modules.mislabeled_images.mislabeled_resources import *

def train_correction_model(dataset: ReiformICDataSet, edge_size: int) -> nn.Module:

    classes = len(dataset.classes())

    model = SmallAutoNet(insize, edge_size, classes)
    dataloader = dataset.get_dataloader(insize, edge_size, CORRECTION_MODEL_BATCH_SIZE)
    train_conv_net(model, dataloader, multiclass_model_loss, 
                    get_optimizer(model), epochs)

    return model

def train_correction_model_ensemble(dataset : ReiformICDataSet) -> List[nn.Module]:
    
    models : List[nn.Module] = []
    
    for edge_size in edgesizes:
        model = train_correction_model(dataset, edge_size)
        models.append(model)

    return models

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


def predict_correct_label_ensemble(dataset : ReiformICDataSet, models: List[nn.Module]) -> ReiformICDataSet:

    for edge_size, model in zip(edgesizes, models):
        dataset = predict_correct_label(dataset, model, edge_size)

    return dataset

def evaluate_correction_confidence(starting_label: int, predictions: NDArray) -> Tuple[bool, int]:
    # This will be complex: We will evaluate the combination of all label confidences
    # from all models in the ensemble. Think about best way to do this.
    # Possible outcomes: remove, correct, remain

    # predictions => (P, C) ; P is number of predictions, C is number of classes

    # Compute the averages
    average = np.mean(predictions, axis=0)
    max_avg = np.argmax(average)

    # Compute the freq that something is the max in the array
    max_freq = [0] * average.size
    max_indices = np.argmax(predictions, axis=1)
    for val in max_indices:
        max_freq[int(val)] = max_freq[int(val)] + 1
    max_max = max_freq.index(max(max_freq))

    # Compute the freq of indices being > 0.5
    predictions[predictions < 0.5] = 0
    predictions[predictions > 0] = 1
    thresholds = np.sum(predictions, axis=0)
    max_thresh = np.argmax(thresholds)

    # Find consensus
    if   max_avg == max_max and max_avg == max_thresh:
        return True, int(max_avg)
    elif max_max == max_thresh and starting_label == max_max:
        return True, int(max_max)
    elif max_thresh == max_avg and starting_label == max_thresh:
        return True, int(max_thresh)

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
        models : List[nn.Module] = train_correction_model_ensemble(incl)

        # Classify contents of 'to_correct' using new models
        to_correct = predict_correct_label_ensemble(to_correct, models)

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
    for iter in range(iterations):
        ReiformInfo("Iteration: {} / {}".format(iter+1, iterations))

        corrected, dropped = monte_carlo_label_correction(monte_carlo_simulations, inliers, outliers)
        all_corrected.merge_in(corrected)

        if outliers.file_count() == 0:
            inliers.set_minus(corrected)
            inliers.set_minus(dropped)

        inliers.merge_in(corrected)

        if outliers == dropped:
            ReiformInfo("Early termination of iterative label correction: outliers == dropped (no improvement)")
            break

        outliers = dropped

    ReiformInfo("Full Monte Carlo timing: {} seconds".format(round(time.time() - start) ))
    return inliers, outliers, corrected