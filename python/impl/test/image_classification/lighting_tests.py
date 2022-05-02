from impl.services.modules.lighting_correction.lighting_datasets import *
from impl.services.modules.lighting_correction.lighting_models import *
from impl.services.modules.lighting_correction.lighting_utils import *
from impl.services.modules.lighting_correction.correction import *
from .test_utils import dataset_evaluation, train_model_for_eval


def test_detection_model(model_path, path, all_dark, all_light, bare_model_class):
    
    ds = dataset_from_path(path)
    
    sizes, _ = max_sizes(ds)
    edge_size = min(256, closest_power_of_2(max(sizes[:2])))
    channels = sizes[2]
    
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingDetection(ds, transform, all_dark, all_light)

    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=1, shuffle=False, num_workers=1)

    model = bare_model_class(channels, edge_size)
    model.load_state_dict(torch.load(model_path))

    eval_model(dataloader, model)

def test_correction_model(model_path, path):
    edge_size = 128
    ds = dataset_from_path(path)
    
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size),
            transforms.ToTensor()
        ])

    model = LightingCorrectionNet(edge_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Get a dataloader that has intentional lighting mistakes 
    train_ds, test_ds = ds.split(0.9)

    correct_train_dl = torch.utils.data.DataLoader(DatasetForLightingCorrectionTest(train_ds, transform),
                                           batch_size=128, shuffle=False, num_workers=1)
    # This will be resolved with the issue #123
    test_temp_path : str = "./impl/test/working/temp"

    model = model.to(device)
    model.eval()

    for image, label, name in correct_train_dl:
        image = image.to(device)
        pred = model(image)
        for i in range(len(label)):
            new_path = "{}/{}/{}".format(test_temp_path, label[i], name.split("/")[-1])
            im = transforms.ToPILImage()(pred[i])
            im.save(new_path)

    corrected_train_ds = dataset_from_path(test_temp_path)

    # Train a model using each, evaluate on the same dataset, and return results
    corrected_scores = dataset_evaluation(corrected_train_ds, test_ds)
    
    # Now raw
    sizes, max_ = max_sizes(train_ds)
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

def test(val_path : str, correction_model_path : str, detection_model_path):

    # Test different Detection models
    model_type = LightingDetectorSparse
    test_detection_model(detection_model_path, val_path, False, False, model_type )
    
    # Test Correction
    test_correction_model(correction_model_path, val_path)
    
if __name__ == "__main__":
    data_path = sys.argv[1]
    corr_model_path = sys.argv[2]
    det_model_path = sys.argv[3]

    test(data_path,
        corr_model_path,
        det_model_path)