from pathlib import Path
from impl.services.image_classification.resources import *
from impl.services.modules.mislabeled_images.detection import *
from impl.services.modules.mislabeled_images.correction import *
from impl.services.modules.mislabeled_images.report_generation import *
from impl.services.modules.core.embeddings.pretrained_embedding import *
from impl.test.image_classification.test_utils import dataset_evaluation

logging.getLogger('PIL').setLevel(logging.WARNING)

def test_vae_projection(data : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet]:

    data = data.mislabel(5)
    start = time.time()
    channels_in, edge_size, dataloader, projection_dataloader = create_dataloaders(data)
    results : ReiformICDataSet = vae_projection(data, EMBEDDING_DIM_SIZE, channels_in, edge_size, dataloader, projection_dataloader)
    ReiformInfo("Time for latent projection: {}".format(time.time() - start))
    
    inliers, outliers = find_outlier_consensus(results)   

    return inliers, outliers

def test_pretrained_projection_detection(data : ReiformICDataSet) -> ReiformICDataSet:

    data = data.mislabel(5)

    return run_pretrained_detection(data)

def run_pretrained_detection(data : ReiformICDataSet):
    # Run embedding code
    path_to_embeddings = LOCAL_EMBEDDING_PATH
    get_dataset_embedding(data, path_to_embeddings)

    inliers, outliers = find_outlier_consensus(data)

    return inliers, outliers


def run_label_correction(inliers : ReiformICDataSet, 
                          outliers : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet, ReiformICDataSet]:
    inliers, outliers, corrected = iterative_reinjection_label_correction(25, inliers, outliers)

    return inliers, outliers, corrected

def test_label_correction(dataset : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    # Takes in a dataset and returns the corrected data to write out
    inliers, outliers = dataset.split(0.9)
    outliers = outliers.mislabel(101)

    inl, new_out, corrected = run_label_correction(inliers, outliers)

    ReiformInfo("Diff: {}".format(inl.count_differences(dataset)))
    ReiformInfo("Outliers Remaining: {}".format(new_out.file_count()))

    return new_out, corrected

def run_tests(data_path=None, results_path=None):

    random_seed = 433
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # These should be the actual dirs in git with the tests files
    # data_path : str = "./python/impl/test/test_data_mnist"
    # results_path : str = "./python/impl/test/test_results"
    if data_path is None:
        data_path : str = "python/impl/test/test_data_cifar"
        data_path : str = "python/impl/test/test_data_color"
    if results_path is None:
        results_path : str = "python/impl/test/test_results"

    do_dataset_evaluation = False
    do_test_detection = True
    do_test_correction = False

    dataset : ReiformICDataSet = dataset_from_path(data_path)

    if do_dataset_evaluation:

        train_ds, test_ds = dataset.split(0.7)

        train_ds = train_ds.mislabel(5)

        ReiformInfo("Data split and mislabeled. Detection starting.")
        inliers, outliers = run_pretrained_detection(train_ds)
        ReiformInfo("Correction starting.")
        inliers, outliers, corrected = run_label_correction(inliers, outliers)

        ReiformInfo("Mislabeled evaluation starting.")
        raw_scores = dataset_evaluation(train_ds, test_ds)
        ReiformInfo("Corrected evaluation starting.")
        corrected_scores = dataset_evaluation(inliers, test_ds)

        ReiformInfo("Raw Scores       : {}".format(str(raw_scores)))
        ReiformInfo("Corrected Scores : {}".format(str(corrected_scores)))

    if do_test_detection:

        # Test that this runs - good in case we need it for custom embeddings
        # inliers, outliers = test_vae_projection(dataset)

        # "Real" tests start here
        inliers, outliers = test_pretrained_projection_detection(dataset)
        
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

    if len(sys.argv) > 1:
        data_path=sys.argv[1]
    if len(sys.argv) > 2:
        res_path=sys.argv[2]

    run_tests(data_path, res_path)