from impl.services.modules.lighting_correction.lighting_datasets import *
from impl.services.modules.lighting_correction.lighting_models import *
from impl.services.modules.lighting_correction.lighting_utils import *
from impl.services.modules.lighting_correction.correction import *
from .test_utils import dataset_evaluation, train_model_for_eval

def test_train_detection(path : str, val_path: str):
    learning_rate = 0.00005
    momentum = 0.94
    epochs = 5
    edge_size = 128
    batch_size = 32
    model = LightingDetectorSparse()
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-2)

    start = time.time()
    ds = dataset_from_path(path)
    val_ds = dataset_from_path(val_path)
    ReiformInfo("Read in dataset: {}".format(round(time.time() - start)))
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingDetection(ds, transform)
    val_pt_ds = DatasetForLightingDetection(val_ds, transform)

    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = torch.utils.data.DataLoader(val_pt_ds, batch_size=32, shuffle=False, num_workers=1)

    start = time.time()
    model, loss_list = train_detection(model, dataloader, val_dataloader, epochs, optimizer)
    ReiformInfo("Training Duration: {}".format(round(time.time() - start)))

    return model

def test_train_correction(path: str, val_path: str, model_path=None, epoch_start=0):
    learning_rate = 0.00005
    momentum = 0.94
    epochs = 40
    edge_size = 128
    batch_size = 240

    model = LightingCorrectionNet(edge_size)
    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-2)

    start = time.time()
    ds = dataset_from_path(path)
    val_ds = dataset_from_path(val_path)
    ReiformInfo("read in dataset: {}".format(round(time.time() - start)))
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingCorrection(ds, transform)
    val_pt_ds = DatasetForLightingCorrection(val_ds, transform)

    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = torch.utils.data.DataLoader(val_pt_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    start = time.time()
    model, loss_list = train_lighting_correction(model, dataloader, val_dataloader, epochs, optimizer, epoch_start=epoch_start)
    ReiformInfo("Training Duration: {}".format(round(time.time() - start)))

    return model

def test_detection_model(model_path, path, all_dark, all_light, bare_model_class):
    edge_size = 128
    ds = dataset_from_path(path)
    
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingDetection(ds, transform, all_dark, all_light)

    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=1, shuffle=False, num_workers=1)

    model = bare_model_class()
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

def test():
    
    # The paths will be updated in a different PR! (Issues 122/123)
    val_path = "/home/ben/Data/open_images/images/train/"
    
    correction_model_path : str = "checkpoint.pt"
    test_correction_model(correction_model_path, val_path)

    detection_model_path_dark : str = "detection_dark.pt"
    detection_model_path_bright : str = "detection_bright.pt"
    detection_model_path_both : str = "detection_both.pt"

    model_type = LightingDetectorSparse
    test_detection_model(detection_model_path_dark, val_path, True, False, model_type )
    test_detection_model(detection_model_path_bright, val_path, False, True, model_type )
    test_detection_model(detection_model_path_both, val_path, False, False, model_type )

    
if __name__ == "__main__":
    test()