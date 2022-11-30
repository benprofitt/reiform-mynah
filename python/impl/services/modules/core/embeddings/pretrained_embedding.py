import json
from pathlib import Path
from impl.services.modules.core.resources import *
from impl.services.modules.core.vae_auto_net import *
from impl.services.modules.core.vae_models import *
from impl.services.modules.core.reiform_imageclassificationdataset import *
from impl.services.modules.utils.image_utils import closest_power_of_2


def save_embedding_model(model : torch.nn.Module, channels : int, 
               size : int, resize : str,
               mean : List[float], std : List[float],
               latent_size : int, dataset_name : str) -> None:
    # Save the embedding model along with metadata
    json_body = {
        CHANNELS : channels,
        SIZE : size, # input edge size
        RESIZE : resize, # Squeeze / Stretch OR min size
        MEAN : mean,
        STD : std,
        LATENT_SIZE: latent_size,
        NAME: dataset_name
    }

    local_path = get_embedding_path(LOCAL_EMBEDDING_PATH_MOBILENET, channels, size, resize, mean, std, dataset_name)
    Path("/".join(local_path.split("/")[0:-1])).mkdir(exist_ok=True, parents=True)

    model_path = "{}{}".format(local_path, "_model.pt")
    metadata_path = "{}{}".format(local_path, "_metadata.json")

    # Save the model and the json
    torch.save(model.state_dict(), model_path)
    with open(metadata_path, 'w') as fh:
        fh.write(json.dumps(json_body, indent=2))

    return

def get_embedding_path(local_path, channels=None, size=None, resize=None, mean=None, std=None, dataset_name=None):
    
    additions = [channels, size, resize]

    for item in additions:
        if item is None:
            return local_path
        local_path = "{}/{}".format(local_path, item)

    if mean is None:
        return local_path
    local_path = "{}/{}".format(local_path, "_".join([str(m) for m in mean]))

    if std is None:
        return local_path
    local_path = "{}/{}".format(local_path, "_".join(str(m) for m in std))

    if dataset_name is None:
        return local_path
    local_path = "{}/{}".format(local_path, dataset_name)

    return local_path

def load_embedding_model(model_path : str, json_body : Dict[str, Any]) -> EncoderAutoNet:
    
    channels_in = json_body[CHANNELS]
    edge_size = json_body[SIZE]
    latent_size = json_body[LATENT_SIZE]
    
    encoder = VAEAutoNet(channels_in, edge_size, latent_size).encoder

    # Load the state dict
    encoder = load_pt_model(encoder, model_path)
    encoder.eval()
    
    return encoder

def train_embedding(dataset : ReiformICDataSet, transformation : transforms.Compose, 
                    edge_size : int, channels : int, latent_size : int):

    # Training a single embedding for a dataset and size
    batch_size=min(int(BASE_EMBEDDING_MODEL_BATCH_SIZE * 1024**2/edge_size**2), MAX_EMBEDDING_MODEL_BATCH_SIZE)
    dataloader = dataset.get_dataloader(channels, edge_size=edge_size, batch_size=batch_size, transformation=transformation)
    proj_dataloader = dataset.get_dataloader(channels, edge_size=edge_size, batch_size=batch_size, transformation=transformation)

    vae = train_encoder_vae(latent_size, channels, edge_size, 
                            dataloader, proj_dataloader)

    return vae.encoder

def train_embedding_sizes(dataset : ReiformICDataSet, channels : int, 
                          resize : str, mean : List[float], std : List[float],
                          latent_size : int, dataset_name : str):
    # Here we will train embeddings for various sizes
    # TODO : Move these sizes to some config file/resources file
    for edge_size in [1024, 32, 64, 128, 256, 512]:
        local_path = get_embedding_path(LOCAL_EMBEDDING_PATH_MOBILENET, channels, edge_size, resize, mean, std, dataset_name)

        if os.path.isfile(local_path):
            ReiformWarning("You are trying to train a model that already exists: {}, {}, {}, {}. Skipping.".format(channels, edge_size, resize, dataset_name))
            continue

        resize_str = "transforms.Resize" + ("({})".format(edge_size) if resize == "min_size" else "(({}, {}))".format(edge_size, edge_size))

        transformation = transforms.Compose([
            eval(resize_str),
            transforms.RandomCrop(edge_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        encoder = train_embedding(dataset, transformation, edge_size, channels, latent_size)

        save_embedding_model(encoder, channels, edge_size, resize, mean, 
                             std, latent_size, dataset_name)

def train_embedding_for_dataset(dataset : ReiformICDataSet, name : str):
    # Here we will train the embeddings for various datasets

    sizes = dataset.find_max_image_dims()
    channels = sizes[2]
    latent_size = EMBEDDING_DIM_SIZE

    for resize in ["min_size", "stretch"]:
        train_embedding_sizes(dataset, channels, resize, dataset.get_mean(), 
                              dataset.get_std_dev(), latent_size, name)

def train_encoder_vae(latent_size : int, channels_in : int, edge_size : int, 
                      dataloader, projection_dataloader):
    # Make model and optimizer
    vae = VAEAutoNet(channels_in, edge_size, latent_size)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=0.001, weight_decay=1e-2)

    # Train the VAE
    vae, _ = train_projection_separation_vae(vae, dataloader, projection_dataloader, 
                                             VAE_PROJECTION_TRAINING_EPOCHS, optimizer)
    return vae

def create_dataloaders(data : ReiformICDataSet):
    sizes = data.find_max_image_dims()
    max_size = max(sizes)
    channels_in : int = sizes[2]

    closest_size = closest_power_of_2(max_size)

    ReiformInfo("Size of input: {}".format(closest_size))
    # Make dataloader from dataset
    
    dataloader = data.get_dataloader(channels_in, closest_size, 
                                     CORRECTION_MODEL_BATCH_SIZE)
    proj_dataloader = data.get_dataloader(channels_in, closest_size, 
                                          CORRECTION_MODEL_BATCH_SIZE)

    ReiformInfo("Dataloader Created")
    return channels_in,closest_size,dataloader,proj_dataloader