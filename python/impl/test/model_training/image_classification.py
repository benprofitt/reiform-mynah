import sys
from typing import Any, Dict
from impl.services.image_classification.model_training import TrainingJob
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICDataSet, dataset_from_path
from impl.services.modules.utils.model_utils import load_reiform_model
from impl.services.modules.utils.progress_logger import TestLogger
from impl.services.modules.utils.reiform_exceptions import ReiformInfo


def test_training_job(request):

    training_job : TrainingJob = TrainingJob(request)
    
    training_results : Dict[str, Any] = training_job.run_processing_job(TestLogger())

    ReiformInfo(training_results)
    return training_results

def test_load_saved_model(request : Dict[str, Any], dataset : ReiformICDataSet):

    model = load_reiform_model(request["config_params"]["model_save_path"], request["config_params"]["model"], len(dataset.classes()))

    ReiformInfo(model)
    
def calc_training_analytics(results : Dict[str, Any]):
    test_loss_avg = []

    all_test_loss = results["all_file_test_loss"]

    for ep in all_test_loss:
        test_loss_avg.append(sum(ep)/len(ep))
    
    last_test_ep = all_test_loss[-1]
    average_final_loss = test_loss_avg[-1]
    worst_test_losses = []

    for i, fileid in enumerate(results["ordered_fileids"]):
        if last_test_ep[i] > average_final_loss:
            worst_test_losses.append((last_test_ep[i], fileid))

    return worst_test_losses, test_loss_avg, results["avg_training_loss"]

def test_suite(dataset_path : str):

    dataset : ReiformICDataSet = dataset_from_path(dataset_path)

    request : Dict[str, Any] = {
        "config_params": {
        "model" : "resnet50",
        "min_epochs" : 3,
        "max_epochs" : 50,
        "loss_epsilon" : 0.001,
        "batch_size" : 64,
        "train_test_split" : 0.9,
        "model_save_path" : "impl/test/temp_results/test_model.pt"
        },
        "dataset": dataset.to_json()
    }

    results = test_training_job(request)

    test_load_saved_model(request, dataset)

    files_with_worst_losses, avg_test_losses, avg_training_losses = calc_training_analytics(results)

    ReiformInfo(files_with_worst_losses)
    ReiformInfo(avg_test_losses)
    ReiformInfo(avg_training_losses)

def run_IC_training_tests():
    path = sys.argv[1]
    test_suite(path)