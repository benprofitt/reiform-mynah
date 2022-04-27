from impl.services.modules.lighting_correction.lighting_datasets import *
from impl.services.modules.lighting_correction.lighting_models import *
from impl.services.modules.lighting_correction.lighting_utils import *
from impl.services.modules.lighting_correction.correction import *

def test_bright(im):
    plt.imshow(make_bright(im))
    plt.show()

def test_dark(im):
    plt.imshow(make_dark(im))
    plt.show()

def test_train_detection(path : str, val_path: str):
    learning_rate = 0.00005
    momentum = 0.94
    epochs = 5
    edge_size = 128
    batch_size = 32
    model = LightingDetectorSparse()
    
    print(count_parameters(model))

    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-2)

    start = time.time()
    ds = dataset_from_path(path)
    val_ds = dataset_from_path(val_path)
    print("read in dataset: {}".format(round(time.time() - start)))
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingDetection(ds, transform)
    val_pt_ds = DatasetForLightingDetection(val_ds, transform)

    print("pt dataset")
    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = torch.utils.data.DataLoader(val_pt_ds, batch_size=32, shuffle=False, num_workers=1)

    print("pt dataloader")
    start = time.time()
    model, loss_list = train_detection(model, dataloader, val_dataloader, epochs, optimizer)
    print("Training Duration: {}".format(round(time.time() - start)))

    return model

def test_train_detection_tensor(path : str):
    learning_rate = 0.001
    momentum = 0.95
    epochs = 10
    edge_size = 128
    batch_size = 32
    model = LightingDetectorSparse()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-2)

    ds = dataset_from_path(path)
    print("read in dataset")
    

    pt_ds = DatasetForLightingDetectionTensors(ds)

    print("pt dataset")
    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

    print("pt dataloader")
    start = time.time()
    model, loss_list = train_detection(model, dataloader, dataloader, epochs, optimizer)
    print("Training Duration: {}".format(time.time() - start))

    return model

def test_detection_model(model_path, path, all_dark, all_light, bare_model_class):
    edge_size = 128
    ds = dataset_from_path(path)
    print("read in dataset")
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingDetection(ds, transform, all_dark, all_light)

    print("pt dataset")
    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=1, shuffle=False, num_workers=1)

    model = bare_model_class()
    model.load_state_dict(torch.load(model_path))

    eval_model(dataloader, model)

def write_out_tensor(path):

    edge_size = 128
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size),
            transforms.ToTensor()
        ])

    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            full_name = "{}/{}".format(path, file)
            tensor = transform(Image.open(full_name))
            torch.save(tensor, full_name.split(".")[0] + ".pt")

def test_correction_model(model_path, path):
    edge_size = 128
    ds = dataset_from_path(path)
    print("read in dataset")
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingCorrection(ds, transform)

    print("pt dataset")
    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=1, shuffle=False, num_workers=1)

    model = LightingCorrectionNet(edge_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    for im, target, label in dataloader:
        target = target.to(device)

        f, axarr = plt.subplots(1,3)
        axarr[0].imshow( im[0].cpu().permute(1, 2, 0).detach().numpy())
        axarr[1].imshow( target[0].cpu().permute(1, 2, 0).detach().numpy())
        recon = model(target)[0].cpu().permute(1, 2, 0).detach().numpy()
        print(recon.shape)
        axarr[2].imshow( recon)

        plt.show()

def test_train_correction(path: str, val_path: str, model_path=None, epoch_start=0):
    learning_rate = 0.00005
    momentum = 0.94
    epochs = 40
    edge_size = 128
    batch_size = 24

    model = LightingCorrectionNet(edge_size)
    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    
    print(count_parameters(model))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-2)

    start = time.time()
    ds = dataset_from_path(path)
    val_ds = dataset_from_path(val_path)
    print("read in dataset: {}".format(round(time.time() - start)))
    transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size)
        ])

    pt_ds = DatasetForLightingCorrection(ds, transform)
    val_pt_ds = DatasetForLightingCorrection(val_ds, transform)

    print("pt dataset")
    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = torch.utils.data.DataLoader(val_pt_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    print("pt dataloader")
    start = time.time()
    model, loss_list = train_lighting_correction(model, dataloader, val_dataloader, epochs, optimizer, epoch_start=epoch_start)
    print("Training Duration: {}".format(round(time.time() - start)))

    return model

def test():
    im_name = "/home/ben/Data/test_images/airport_image.jpg"
    im = Image.open(im_name)

    if False:
        test_dark(im)
        plt.imshow(im)
        plt.show()

        test_bright(im)
    
    # val_path = "/home/ben/Data/open_images/images/train_1/"
    # val_path = "/home/ben/Data/open_images/images/test_testing/"
    val_path = "/home/ben/Data/open_images/images/train_2_testing/"
    val_path = "/home/ben/Data/open_images/images/train/"

    if VERBOSE:
        # path = "/home/ben/Data/open_images/images/train/"
        # model = test_train_detection(path)
        # path = "/home/ben/Data/open_images/tensors/train/"
        # model = test_train_detection_tensor(path)
        print("Verbose")

    path = "/home/ben/Data/open_images/images/train_0/"
    # path = "/home/ben/Data/open_images/images/train_testing/"
    # path = "/home/ben/Data/open_images/images/train/"

    id = "1647840222.653724"
    model_path : str = "checkpoint.pt"
    model = test_train_correction(path, val_path, model_path, epoch_start=24)

    m_path : str = "/home/ben/Data/open_images/images/models/correction_pt_model_{}.pt".format(time.time())

    torch.save(model.state_dict(), m_path)

    test_correction_model(m_path, val_path)

    # model_type = LightingDetectorSparse
    # # test_detection_model(m_path,
    # #                     "/home/ben/Data/open_images/images/train", True, False, model_type )
    # # test_detection_model(m_path,
    # #                     "/home/ben/Data/open_images/images/train", False, True, model_type )
    # test_detection_model(m_path,
    #                     "/home/ben/Data/open_images/images/test_testing", False, False, model_type )

    
if __name__ == "__main__":
    test()