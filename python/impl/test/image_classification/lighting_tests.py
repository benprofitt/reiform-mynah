from impl.services.modules.lighting_correction.lighting_datasets import *
from impl.services.modules.lighting_correction.lighting_models import *
from impl.services.modules.lighting_correction.lighting_utils import *
from impl.services.modules.lighting_correction.correction_models import *
from impl.services.modules.lighting_correction.correction import run_correction_model
from impl.services.modules.lighting_correction.detection import run_detection_models
from impl.services.modules.lighting_correction.pretraining import get_pretrained_path_lighting
from impl.services.modules.utils.data_formatting import load_dataset
from .test_utils import dataset_evaluation, train_model_for_eval


def test_detection_model(models_path, dataset, all_dark, all_light, bare_model_class):
    
    ds = dataset
    sizes = ds.find_max_image_dims()
    edge_size = max(64, min(512, closest_power_of_2(max(sizes[:2]))))
    channels = sizes[2]

    path_to_models = get_pretrained_path_lighting(models_path, channels, edge_size)
    model_files = glob(path_to_models + '/**/*.pt', recursive=True)
    _ = glob(path_to_models + '/**/*.json', recursive=True)
    model_path = model_files[0]
        
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingDetection(ds, transform, all_dark, all_light)

    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=1, shuffle=False, num_workers=1)

    model = bare_model_class(channels, edge_size)
    model.load_state_dict(torch.load(model_path))

    eval_model(dataloader, model)

def test_correction_model(detection_models_path, corr_models_path, dataset : ReiformICDataSet, correction_save_path):
    
    # Get a dataloader that has intentional lighting mistakes 
    train_ds, test_ds = dataset.split(0.9)

    train_ds_to_correct = train_ds.copy()
    _, affected = run_detection_models(train_ds_to_correct, detection_models_path)

    sizes = dataset.find_max_image_dims()
    edge_size = max(64, min(1024, closest_power_of_2(max(sizes[:2]))*2))
    edge_size = (edge_size if edge_size in LIGHTING_CORRECTION_EDGE_SIZES else edge_size*2)
    
    for k, v in affected.items():
        for filename in v:
            file = train_ds_to_correct.get_file_by_uuid(filename)
            file.move_image(correction_save_path)
    
    run_correction_model(corr_models_path, train_ds_to_correct)

    corrected_train_ds = train_ds_to_correct

    # Train a model using each, evaluate on the same dataset, and return results
    corrected_scores = dataset_evaluation(corrected_train_ds, test_ds)
    
    # Now raw
    sizes = train_ds.find_max_image_dims()
    max_ = max(sizes)
    edge_size = closest_power_of_2(max_)
    batch_size = 256
    epochs = 25
    classes = len(train_ds.classes())

    transform : torchvision.transforms.Compose = transforms.Compose([
        transforms.Resize(edge_size),
        transforms.RandomCrop(edge_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.get_mean(), std=test_ds.get_std_dev())
    ])
    
    test_dl = test_ds.get_dataloader(sizes[2], edge_size, batch_size, transform)

    raw_train_dl = torch.utils.data.DataLoader(DatasetForLightingCorrectionTest(train_ds, transform),
                                           batch_size=128, shuffle=False, num_workers=1)

    precorrection_scores = train_model_for_eval(raw_train_dl, test_dl, sizes, edge_size, epochs, classes)

    ReiformInfo("Scores for model trained on uncorrected data: {}".format(str(precorrection_scores)))
    ReiformInfo("Scores for model trained on corrected data: {}".format(str(corrected_scores)))

def test(dataset_path : str, correction_model_path : str, detection_model_path : str, save_path : str):

    if int(sys.argv[5]):

        dataset = dataset_from_path(dataset_path)
    else:
        dataset = load_dataset(dataset_path)

    # Test different Detection models
    model_type = LightingDetectorSparse
    test_detection_model(detection_model_path, dataset, False, False, model_type)
    
    # Test Correction
    test_correction_model(detection_model_path, correction_model_path, dataset, save_path)
    
if __name__ == "__main__":
    data_path = sys.argv[1]
    corr_model_path = sys.argv[2]
    det_models_path = sys.argv[3]
    save_path = sys.argv[4]

    test(data_path,
        corr_model_path,
        det_models_path,
        save_path)