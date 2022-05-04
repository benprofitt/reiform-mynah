from .lighting_resources import *
from .lighting_models import *
from .lighting_datasets import *
from .correction import LightingCorrectionNet, train_lighting_correction

def train_detection_model(channels : int, edge_size : int, dataloader):
    learning_rate = 0.00005
    momentum = 0.94
    epochs = 5

    model = LightingDetectorSparse(channels, edge_size)
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-2)

    start = time.time()
    model, _ = train_detection(model, dataloader, epochs, optimizer)
    ReiformInfo("Training Duration: {}".format(round(time.time() - start)))

    return model

def get_pretrained_path_detection(local_path : str, channels : int = None, 
                                  edge_size : int = None, dataset_name : str = None):
    additions = [channels, edge_size]

    for item in additions:
        if item is None:
            return local_path
        local_path = "{}/{}".format(local_path, item)

    if dataset_name is None:
        return local_path
    local_path = "{}/{}".format(local_path, dataset_name)

    return local_path

def save_correction_model(model : torch.nn.Module, channels : int, 
               size : int, dataset_name : str) -> None:
    pass

def save_detection_model(model : torch.nn.Module, channels : int, 
               size : int, dataset_name : str) -> None:
    # Save the embedding model along with metadata
    json_body = {
        CHANNELS : channels,
        SIZE : size, # input edge size
        NAME: dataset_name
    }

    local_path = get_pretrained_path_detection(LOCAL_PRETRAINED_PATH_LIGHT_DETECTION, channels, size, dataset_name)
    Path("/".join(local_path.split("/")[0:-1])).mkdir(exist_ok=True)

    model_path = "{}{}".format(local_path, "_model.pt")
    metadata_path = "{}{}".format(local_path, "_metadata.json")

    # Save the model and the json
    torch.save(model.state_dict(), model_path)
    with open(metadata_path, 'w') as fh:
        fh.write(json.dumps(json_body, indent=2))

    return

def load_pretrained_model(model_path : str, json_body : Dict[str, Any], model_type : Any):
    channels_in = json_body[CHANNELS]
    edge_size = json_body[SIZE]
    
    model = model_type(channels_in, edge_size)

    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def pretrain_model(dataset : ReiformICDataSet, transformation : transforms.Compose, 
                    edge_size : int, channels : int, train_model_func : Callable):

    # Training a single model for a dataset and size
    dataloader = dataset.get_dataloader(channels, edge_size=edge_size, 
                                        batch_size=CORRECTION_MODEL_BATCH_SIZE, 
                                        transformation=transformation)

    model = train_model_func(channels, edge_size, dataloader)

    return model

def pretrain_model_sizes(dataset : ReiformICDataSet, channels : int,
                          dataset_name : str, train_model_func : Callable, 
                          save_func : Callable,
                          resize : str, mean : List[float] = None, std : List[float] = None):
    # Here we will train embeddings for various sizes
    # TODO : Move these sizes to some config file/resources file
    for edge_size in [32, 64, 128, 256]:

        resize_str = "transforms.Resize" + ("({})".format(edge_size) if resize == "min_size" else "(({}, {}))".format(edge_size, edge_size))

        transformation = transforms.Compose([
            eval(resize_str),
            transforms.RandomCrop(edge_size),
            transforms.ToTensor()
        ])

        if mean is not None:
            transformation = transforms.Compose([
                transformation,
                transforms.Normalize(mean=mean, std=std)
            ])

        model = pretrain_model(dataset, transformation, edge_size, channels, train_model_func)

        save_func(model, channels, edge_size, dataset_name, resize, mean, std)

def train_detection_for_dataset(path_to_dataset : str, name : str):
    # Here we will train the embeddings for various datasets

    dataset = dataset_from_path(path_to_dataset)
    sizes, _ = max_sizes(dataset)
    channels = sizes[2]

    resize = "stretch"
    pretrain_model_sizes(dataset, channels, name, train_detection_model, save_detection_model, resize)

def train_correction_for_dataset(path: str, val_path: str, model_path=None, epoch_start=0):
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

    dataloader = torch.utils.data.DataLoader(pt_ds, batch_size=batch_size, shuffle=True, num_workers=WORKERS)
    val_dataloader = torch.utils.data.DataLoader(val_pt_ds, batch_size=batch_size, shuffle=False, num_workers=WORKERS)

    start = time.time()
    model, loss_list = train_lighting_correction(model, dataloader, val_dataloader, epochs, optimizer, epoch_start=epoch_start)
    ReiformInfo("Training Duration: {}".format(round(time.time() - start)))

    return model

if __name__ == "__main__":
    task = sys.argv[1]
    path = sys.argv[2]
    name = sys.argv[3]

    if task == "detection":
        train_detection_for_dataset(path, name)
    else:
        # TODO in 123
        # train_correction_for_dataset()
        pass