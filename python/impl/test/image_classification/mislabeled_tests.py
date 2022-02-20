from pathlib import Path
from python.impl.services.modules.mislabeled_images.mislabeled_resources import *
from python.impl.services.modules.mislabeled_images.detection import *
from python.impl.services.modules.mislabeled_images.correction import *
from python.impl.services.modules.mislabeled_images.report_generation import *


def test_projection(data : ReiformICDataSet) -> ReiformICDataSet:
    try:
        start = time.time()
        results : ReiformICDataSet = vae_projection(data, 2)
        print("Time for latent projection: {}".format(time.time() - start))
    except:
        print("Failure at test_projection")
        return ReiformICDataSet([])
    return results

def test_detection(results : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    inliers, outliers = find_outliers_isolation_forest(results, 0.2)
    return inliers, outliers

def test_2D_report_projection(inliers : ReiformICDataSet, 
                              outliers : ReiformICDataSet) -> Tuple[ReiformICDataSet, 
                                                                  ReiformICDataSet]:
    inliers, outliers = projection_2D_from_outlier_projection_one_class(inliers, outliers)
    return inliers, outliers

def test_label_correction(inliers : ReiformICDataSet, 
                          outliers : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet, ReiformICDataSet]:
    inliers, outliers, corrected = iterative_reinjection_label_correction(1, inliers, outliers)

    return inliers, outliers, corrected

def run_tests():

    random_seed = 11
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    data_path : str = "./impl/test/test_data_mnist"
    results_path : str = "./impl/test/test_results"
    dataset : ReiformICDataSet = dataset_from_path(data_path)

    results : ReiformICDataSet = test_projection(dataset)

    print(results.file_count())

    inliers, outliers = test_detection(results)

    print(inliers.file_count())
    print(outliers.file_count())
    
    # inliers, outliers = test_2D_report_projection(inliers, outliers)
    plot_in_2D(list((inliers, outliers)), PROJECTION_LABEL)

    inliers, outliers, corrected = test_label_correction(inliers, outliers)

    for prefix, ds in zip(["in", "out"], [corrected, outliers]):

        for c in ds.classes():
            new_path : str = "{}/{}/{}/".format(results_path, prefix, c)
            Path(new_path).mkdir(parents=True, exist_ok=True)
            for name, file in ds.get_items(c):
                original : str = file.get_original_class()
                new_filename : str = "{}/{}/{}/{}_{}".format(results_path, prefix, c, original, name.split("/")[-1])
                Image.open(name).save(new_filename)


if __name__ == '__main__':
    run_tests()