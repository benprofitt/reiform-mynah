from impl.services.modules.class_splitting.detection import *
from impl.services.modules.class_splitting.correction import *
from impl.services.modules.core.embeddings.latent_projection import get_dataset_embedding
from impl.services.modules.core.reiform_imageclassificationdataset import dataset_from_path
from impl.test.image_classification.test_utils import dataset_evaluation

def test_splitting_detection(dataset : ReiformICDataSet, 
                             groups_to_combine : List[List[str]]):
    # Test the detection methods for class splitting

    # Combine a few pairs/groups of classes - given by the user
    for g in groups_to_combine:
        dataset.combine_classes(g)

    # Run embedding code
    path_to_embeddings = LOCAL_EMBEDDING_PATH
    get_dataset_embedding(dataset, path_to_embeddings)

    # Run splitting prediction
    split_count, split_predictions = detect_split_need(dataset)

    check_split_counts(dataset, groups_to_combine, split_count, split_predictions)
    ReiformInfo("Class Split Detection Evaluation Complete")


def check_split_counts(dataset, groups_to_combine, split_count, split_predictions):
    # Check if counts match
    for g in groups_to_combine:
        main_class = g[0]
        if main_class not in split_count:
            ReiformInfo("Class group {} not re-split".format(str(g)))
            continue
        else:
            ReiformInfo("Group Size: {} <=> {} :Split Size".format(str(len(g)), 
                                                                   split_count[main_class]))

        # The check what's in each split
        split = split_predictions[main_class]
        ReiformInfo("Looking at split for {}: {}".format(main_class, str(g)))
        for i, cluster in enumerate(split):
            counts = {}
            for uuid in cluster:
                file = dataset.get_file(main_class, uuid)
                if file.original_class not in counts:
                    counts[file.original_class] = 0
                counts[file.original_class] += 1

            ReiformInfo("\tCluster {}: {}".format(i, str(counts)))


def test_splitting_correction(dataset : ReiformICDataSet,
                              groups_to_combine : List[List[str]]):

    # Combine a few pairs/groups of classes - given by the user
    for g in groups_to_combine:
        dataset.combine_classes(g)

    # Train/Eval model on the dataset with combined classes
    train_ds, test_ds = dataset.split(0.9)
    combined_scores = dataset_evaluation(train_ds, test_ds)

    # Run embedding code
    path_to_embeddings = LOCAL_EMBEDDING_PATH
    get_dataset_embedding(dataset, path_to_embeddings)

    # Run splitting prediction
    main_classes = [g[0] for g in groups_to_combine]
    fixed_dataset, split_predictions = split_dataset(dataset, main_classes)
    
    train_ds, test_ds = fixed_dataset.split(0.9)
    split_scores = dataset_evaluation(train_ds, test_ds)

    ReiformInfo("Scores for grouped data : {}".format(str(combined_scores)))
    ReiformInfo("Scores for split data   : {}".format(str(split_scores)))

    ReiformInfo("Class Split Correction Evaluation Complete")

def test():

    ReiformInfo('Test Script Usage (splitting): <path to dataset> "class1--class4  class2--class3--class5" ')

    dataset_path = sys.argv[1]
    dataset = dataset_from_path(dataset_path)

    classes_to_group = sys.argv[2]
    groups = classes_to_group.split("  ")
    groups_to_combine = [g.split("--") for g in groups]

    test_splitting_detection(dataset, groups_to_combine)

    dataset = dataset_from_path(dataset_path)
    test_splitting_correction(dataset, groups_to_combine)

if __name__ == "__main__":
    test()