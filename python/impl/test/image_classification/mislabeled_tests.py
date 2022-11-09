from pathlib import Path
from impl.services.image_classification.resources import *
from impl.services.modules.mislabeled_images.detection import *
from impl.services.modules.mislabeled_images.correction import *
from impl.services.modules.mislabeled_images.report_generation import *
from impl.services.modules.core.embeddings.pretrained_embedding import *
from impl.test.image_classification.test_utils import dataset_evaluation, dataset_evaluation_resnet

logging.getLogger('PIL').setLevel(logging.WARNING)

def test_vae_projection(data : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet]:

    data = data.mislabel(5)
    start = time.time()
    channels_in, edge_size, dataloader, projection_dataloader = create_dataloaders(data)
    results : ReiformICDataSet = vae_projection(data, EMBEDDING_DIM_SIZE, channels_in, edge_size, dataloader, projection_dataloader)
    ReiformInfo("Time for latent projection: {}".format(time.time() - start))
    
    inliers, outliers = find_outlier_consensus(results)   

    return inliers, outliers

def test_pretrained_projection_detection(data : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet, int]:

    data, count = data.mislabel(5)
    inliers, outliers = run_pretrained_detection(data)

    return inliers, outliers, count

def run_pretrained_detection(data : ReiformICDataSet):
    # Run embedding code
    path_to_embeddings = LOCAL_EMBEDDING_PATH_DENSENET201
    create_dataset_embedding(data, path_to_embeddings)

    inliers, outliers = find_outlier_consensus(data)

    return inliers, outliers


def run_label_correction(inliers : ReiformICDataSet, 
                          outliers : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet, ReiformICDataSet]:
    inliers, outliers, corrected = iterative_reinjection_label_correction(15, inliers, outliers)

    return inliers, outliers, corrected

def test_label_correction(dataset : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    # Takes in a dataset and returns the corrected data to write out
    path_to_embeddings = LOCAL_EMBEDDING_PATH_DENSENET201
    create_dataset_embedding(dataset, path_to_embeddings)

    inliers, outliers = dataset.split(0.9)
    outliers, count = outliers.mislabel(101)

    inl, new_out, corrected = run_label_correction(inliers, outliers)

    ReiformInfo("Diff: {}".format(inl.count_differences(dataset)))
    ReiformInfo("Outliers Remaining: {}".format(new_out.file_count()))

    return new_out, corrected

def run_tests(data_path=None, results_path=None, test_data_path=None):

    random_seed = 433
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # These are examples of these paths, and should be available in git...
    if data_path is None or results_path is None:
        data_path : str = "impl/integration/test_data_small"
        results_path : str = "impl/integration/temp_results"

    if not (os.path.exists(data_path) and os.path.exists(results_path)):
        raise Exception
    
    if (test_data_path is not None and not os.path.exists(test_data_path)):
        raise Exception

    do_only_detection = False
    do_dataset_evaluation = True
    do_test_detection = False
    do_test_correction = False

    dataset : ReiformICDataSet = dataset_from_path(data_path)

    if do_only_detection:
        # Test a few models (or one model)

        paths = [LOCAL_EMBEDDING_PATH_DENSENET201]
        
        for path_to_embeddings in paths:
            # for percent in [0, 0, 0]:
            for percent in [0, 0, 1, 5, 10, 20]:
                ReiformInfo("Model used: {}".format(path_to_embeddings.split("/")[-1]))
                
                data, count = dataset.mislabel(percent)

                create_dataset_embedding(data, path_to_embeddings)

                inliers, outliers = find_outlier_consensus(data)
                
                ReiformInfo("Total file count : {}".format(len(dataset.all_files())))
                out_size = len(outliers.all_files())
                ReiformInfo("Total detected outlier count : {}".format(out_size))
                true_outliers : int = 0
                class_predicted_correctly_from_embedding : int = 0
                class_predicted_correctly_from_embedding_t_o : int = 0
                file : ReiformICFile
                classes = inliers.classes()
                for file in outliers.all_files():
                    class_pred = file.get_projection(NEW_LABEL_PREDICTION)[0]
                    if file.get_class() != file.get_original_class():
                        true_outliers += 1
                        if classes[class_pred] == file.get_original_class():
                            class_predicted_correctly_from_embedding_t_o += 1
                    # ReiformInfo("Original Class and Predicted Class: {} and {}".format())
                    if classes[class_pred] == file.get_original_class():
                        class_predicted_correctly_from_embedding += 1
                ReiformInfo("Correctly predicted original class of 'outliers': {}".format(class_predicted_correctly_from_embedding/out_size))
                ReiformInfo("Correctly predicted class of true outliers: {}".format(class_predicted_correctly_from_embedding_t_o/(true_outliers+1)))
                ReiformInfo("Actual detected outlier count : {}".format(true_outliers))
                ReiformInfo("Found outlier percentage : {}".format(true_outliers/(count + 1)))
                ReiformInfo("Actual detected outlier percentage : {}\n".format(true_outliers/out_size))

                del data

    if do_dataset_evaluation:

        train_ds : ReiformICDataSet = dataset.copy()

        if test_data_path is None:
            train_ds, test_ds = train_ds.split(0.9)
            train_ds, _ = train_ds.mislabel(5)
        else:
            test_ds : ReiformICDataSet = dataset_from_path(test_data_path)

        ReiformInfo("Data split and mislabeled. Detection starting.")
        inliers, outliers = run_pretrained_detection(train_ds)
        ReiformInfo("Correction starting.")
        inliers, outliers, corrected = run_label_correction(inliers, outliers)

        ReiformInfo("Mislabeled evaluation starting.")
        raw_scores = dataset_evaluation_resnet(train_ds, test_ds)
        ReiformInfo("Corrected evaluation starting.")
        corrected_scores = dataset_evaluation_resnet(inliers, test_ds)

        ReiformInfo("Raw Scores       : {}".format(str(raw_scores)))
        ReiformInfo("Corrected Scores : {}".format(str(corrected_scores)))

    if do_test_detection:

        # Test that this runs - good in case we need it for custom embeddings
        # inliers, outliers = test_vae_projection(dataset)

        # "Real" tests start here
        inliers, outliers, mislabeled_count = test_pretrained_projection_detection(dataset)
        
        ReiformInfo("Total file count : {}".format(len(dataset.all_files())))
        ReiformInfo("Total detected outlier count : {}".format(len(outliers.all_files())))
        true_outliers : int = 0
        file : ReiformICFile
        for file in outliers.all_files():
            if file.get_class() != file.get_original_class():
                true_outliers += 1
        ReiformInfo("Actual detected outlier count : {}".format(true_outliers))
        
        inliers, outliers, corrected = run_label_correction(inliers, outliers)

        ReiformInfo("Diff inliers/dataset: {}".format(inliers.count_differences(dataset)))
        ReiformInfo("Diff dataset/inliers: {}".format(dataset.count_differences(inliers)))

        ReiformInfo("Inliers / Outliers / Corrected counts: {} / {} / {}".format(
                            inliers.file_count(), outliers.file_count(), corrected.file_count()))

        for prefix, ds in zip(["in", "out"], [corrected, outliers]):

            for c in ds.classes():
                new_path : str = "{}/{}/{}/".format(results_path, prefix, c)
                Path(new_path).mkdir(parents=True, exist_ok=True)
                for name, file in ds.get_items(c):
                    original : str = file.get_original_class()
                    new_filename : str = "{}/{}/{}/{}_{}".format(results_path, prefix, c, original, name.split("/")[-1])
                    Image.open(name).save(new_filename)
    
    if do_test_correction:
        
        new_out, corrected = test_label_correction(dataset)

        for prefix, ds in zip(["in", "out"], [corrected, new_out]):

            for c in ds.classes():
                new_path : str = "{}/{}/{}/".format(results_path, prefix, c)
                Path(new_path).mkdir(parents=True, exist_ok=True)
                for name, file in ds.get_items(c):
                    original : str = file.get_original_class()
                    new_filename : str = "{}/{}/{}/{}_{}".format(results_path, prefix, c, original, name.split("/")[-1])
                    Image.open(name).save(new_filename)


if __name__ == '__main__':

    data_path=None
    res_path=None
    test_path=None

    if len(sys.argv) > 1:
        data_path=sys.argv[1]
    if len(sys.argv) > 2:
        res_path=sys.argv[2]
    if len(sys.argv) > 3:
        test_path=sys.argv[3]

    run_tests(data_path, res_path, test_path)