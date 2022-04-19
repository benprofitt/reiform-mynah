from pathlib import Path
from impl.services.image_classification.resources import *
from impl.services.modules.mislabeled_images.detection import *
from impl.services.modules.mislabeled_images.correction import *
from impl.services.modules.mislabeled_images.report_generation import *


def test_projection(data : ReiformICDataSet) -> ReiformICDataSet:
    data = data.mislabel(5)
    start = time.time()
    results : ReiformICDataSet = vae_projection(data, 2)
    print("Time for latent projection: {}".format(time.time() - start))
    try:
        pass
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


def run_label_correction(inliers : ReiformICDataSet, 
                          outliers : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet, ReiformICDataSet]:
    inliers, outliers, corrected = iterative_reinjection_label_correction(1, inliers, outliers)

    return inliers, outliers, corrected

def test_label_correction(dataset : ReiformICDataSet) -> Tuple[ReiformICDataSet, ReiformICDataSet]:
    """ Takes in a dataset and returns the corrected data to write out"""
    inliers, outliers = dataset.split(0.9)
    outliers = outliers.mislabel(101)

    inl, new_out, corrected = run_label_correction(inliers, outliers)

    print("Diff: {}".format(inl.count_differences(dataset)))
    print("Outliers Remaining: {}".format(new_out.file_count()))

    return new_out, corrected

def run_tests():

    random_seed = 12
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    data_path : str = "./python/impl/test/test_data_mnist"
    results_path : str = "./python/impl/test/test_results"
    do_test_detection = True
    do_test_correction = True

    dataset : ReiformICDataSet = dataset_from_path(data_path)


    if do_test_detection:
        results : ReiformICDataSet = test_projection(dataset)

        print(results.file_count())

        inliers, outliers = test_detection(results)

        print(inliers.file_count())
        print(outliers.file_count())
        
        # inliers, outliers = test_2D_report_projection(inliers, outliers)
        plot_in_2D(list((inliers, outliers)), PROJECTION_LABEL)

        inliers, outliers, corrected = run_label_correction(inliers, outliers)

        print("Diff: {}".format(inliers.count_differences(dataset)))

        for prefix, ds in zip(["in", "out"], [corrected, outliers]):

            for c in ds.classes():
                new_path : str = "{}/{}/{}/".format(results_path, prefix, c)
                Path(new_path).mkdir(parents=True, exist_ok=True)
                for name, file in ds.get_items(c):
                    original : str = file.get_original_class()
                    new_filename : str = "{}/{}/{}/{}_{}".format(results_path, prefix, c, original, name.split("/")[-1])
                    Image.open(name).save(new_filename)
    
    elif do_test_correction:
        
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
    run_tests()