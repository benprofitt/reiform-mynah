import uuid
from impl.services.modules.core.resources import *
from impl.services.modules.core.vae_auto_net import *
from impl.services.modules.core.vae_models import *
from impl.services.modules.core.reiform_imageclassificationdataset import *
from .pretrained_embedding import *
import pickle


def pretrained_projection(data : ReiformICDataSet, model : torch.nn.Module, 
                          preprocess : transforms.Compose, label : str) -> ReiformICDataSet:
    
    dataloader = data.get_dataloader_with_names(preprocess, CORRECTION_MODEL_BATCH_SIZE)

    return trained_model_projection(data, model, dataloader, label)

def trained_model_projection(data : ReiformICDataSet, model : nn.Module, 
                             dataloader : torch.utils.data.DataLoader, 
                             projection_label : str = PROJECTION_LABEL) -> ReiformICDataSet:
    
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for batch, label, names in dataloader:
            batch = batch.to(device)
            
            projection = model(batch)
            projection = projection.to("cpu").numpy()
            
            label = label.to("cpu").numpy()
            
            for idx in range(len(names)):
                l : str = data.classes()[label[idx]]
                
                file : ReiformICFile = data.get_file(l, names[idx])
                file.add_projection(projection_label, projection[idx])

    return data

def vae_projection(data : ReiformICDataSet, latent_size : int, 
                   channels_in : int, edge_size : int, 
                   dataloader : torch.utils.data.DataLoader,
                   projection_dataloader : torch.utils.data.DataLoader) -> ReiformICDataSet:

    vae = train_encoder_vae(latent_size, channels_in, edge_size, dataloader, 
                            projection_dataloader)

    # pass to 'pretrained projection' 
    vae.eval()
    results : ReiformICDataSet = trained_model_projection(data, vae.encoder, dataloader)

    # return results
    return results

from torchvision.models.feature_extraction import create_feature_extractor # type: ignore

def create_dataset_embedding_reduce(dataset : ReiformICDataSet, path_to_embedding_model : str):
    
    # Get the model file and process dataset. 
    # Add the projection to the dataset. 
    perform_dataset_embedding(dataset, path_to_embedding_model)

    # Use umap_red to reduce to fewer dimensions. (for outliers)
    # Finally, reduce again to 2D (for reporting)
    perform_embedding_reduction(dataset, path_to_embedding_model)

def perform_dataset_embedding(dataset, path_to_embedding_model):
    start = time.time()

    # Load a model from the saved model
    batch_size = BASE_EMBEDDING_MODEL_BATCH_SIZE
    if "mobilenet" in path_to_embedding_model:
        model_ = torchvision.models.mobilenet_v2()
        
        batch_size *= 5

        return_nodes = {
            "features": "embedding"
        }
    elif "inception" in path_to_embedding_model:
        model_ = torchvision.models.inception_v3(init_weights=False)

        batch_size *= 3

        return_nodes = {
            "Mixed_7c": "embedding"
        }
        
    elif "densenet" in path_to_embedding_model:
        model_name = path_to_embedding_model.split("/")[-1].split("-")[0]
        model_ = eval("torchvision.models.{}()".format(model_name))

        num = int(model_name.split("net")[-1])
        # This is just what fits in mem on the GPU.
        batch_size = BASE_EMBEDDING_MODEL_BATCH_SIZE

        return_nodes = {
            "features" : "embedding"
        }

    else:
        # Resnets
        model_name = path_to_embedding_model.split("/")[-1].split("-")[0]
        model_ = eval("torchvision.models.{}()".format(model_name))
        
        num = int(model_name.split("net")[-1])
        batch_size *= 152//num

        return_nodes = {
            "layer4": "embedding"
        }

    model_ = load_pt_model(model_, path_to_embedding_model)
    model_.to(device)
    model_.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_ = create_feature_extractor(model_, eval_return_nodes=return_nodes, train_return_nodes=return_nodes)

    edge_size = max(dataset.find_max_image_dims())
    edge_size=min(256, max(MINIMUM_EDGE_SIZE_EMBEDDINGS, edge_size))
    batch_size = int((224/edge_size)*batch_size)
    pt_dl = dataset.get_dataloader(3, batch_size=batch_size, edge_size=edge_size)

    embeddings : List[NDArray] = []
    file_names : List[str] = []

    for images, labels, name_batch in pt_dl:
        images = images.to(device)

        pred = model_(images)["embedding"]
        x = nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        x = torch.flatten(x, 1)
        x = torch.sigmoid(x)

        file_names += name_batch
        for v in x.to("cpu").detach().numpy():
            embeddings.append(v)

    model_.to("cpu")
    del model_

    ReiformInfo("Time for initial embedding inference: {}".format(time.time() - start))

    for i, name in enumerate(file_names):
        file = dataset.get_file_by_name(name)
        file.add_projection(PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION, embeddings[i])

def perform_embedding_reduction(dataset, path_to_embedding_model):
    embeddings = []
    names : List[Tuple[str, str]] = []

    embeddings_by_class : Dict[str, List[NDArray]] = {}
    names_by_class : Dict[str, List[str]] = {}

    for c in dataset.classes():
        embeddings_by_class[c] = []
        names_by_class[c] = []
        for name, file in dataset.get_items(c):
            embeddings.append(file.get_projection(PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION))
            names.append((c, name))

            embeddings_by_class[c].append(file.get_projection(PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION))
            names_by_class[c].append(name)
    
    start = time.time()

    for c in dataset.classes():
        # Per-class embedding reduction -> used for class splitting
        umap_red = cuml.UMAP(n_components=EMBEDDING_DIM_SIZE)
        reduced_embeddings = umap_red.fit_transform(np.array(embeddings_by_class[c]))
        for i, name in enumerate(names_by_class[c]):
            dataset.get_file(c, name).add_projection(PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS, reduced_embeddings[i])

        umap_red = cuml.UMAP()
        reduced_embeddings = umap_red.fit_transform(np.array(embeddings_by_class[c]))
        for i, name in enumerate(names_by_class[c]):
            dataset.get_file(c, name).add_projection(PROJECTION_LABEL_2D_PER_CLASS, reduced_embeddings[i])

    ReiformInfo("Time for per-class reduction: {}".format(time.time() - start))
    start = time.time()

    # Entire dataset embedding reduction -> used for outlier detection
    umap_red = cuml.UMAP(n_components=EMBEDDING_DIM_SIZE)
    reduced_embeddings = umap_red.fit_transform(np.array(embeddings))
    ReiformInfo("Time for 31-D dataset-level reduction: {}".format(time.time() - start))

    for i, file in enumerate(names):
        dataset.get_file(file[0], file[1]).add_projection(PROJECTION_LABEL_REDUCED_EMBEDDING, reduced_embeddings[i])

    # 2D projections -> Used to show user what's up with these embeddings
    umap_red = cuml.UMAP()
    reduced_embeddings = umap_red.fit_transform(np.array(embeddings))

    for i, file in enumerate(names):
        dataset.get_file(file[0], file[1]).add_projection(PROJECTION_LABEL_2D, reduced_embeddings[i])

    save_embedding_model(umap_red, dataset, PROJECTION_LABEL_REDUCED_EMBEDDING, path_to_embedding_model)

    ReiformInfo("Time for dataset-level reduction: {}".format(time.time() - start))

class EmbeddingReducer():

    def __init__(self, path_to_embedding_model : str, path_to_reducer: str, reduction_label : str) -> None:
        self.path_to_embedding_model = path_to_embedding_model
        self.path_to_reducer = path_to_reducer
        self.reduction_label = reduction_label

    def perform_reduction(self, dataset : ReiformICDataSet):
        reducer : cuml.UMAP = load_umap_model(self.path_to_reducer)
        
        perform_dataset_embedding(dataset, self.path_to_embedding_model)

        embeddings, names = dataset.get_embeddings_from_dataset(PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION)

        reduced_embeddings = reducer.transform(embeddings)

        for i, name in enumerate(names):
            file = dataset.get_file_by_name(name)
            file.add_projection(self.reduction_label, reduced_embeddings[i])

def save_embedding_model(model : cuml.UMAP, dataset : ReiformICDataSet, projection_label: str, path_to_embed_model: str):
    # Saves the embedding model and it's relationship to the dataset

    uuid_path = "{}{}.pkl".format(LOCAL_TMP_MODEL_FOLDER, uuid.uuid4())

    save_umap_model(model, uuid_path)

    dataset.add_reducer_metadata(projection_label, uuid_path, path_to_embed_model)

    return

def save_umap_model(reducer : cuml.UMAP, path : str):

    # Save the trained UMAP model to a file
    with open(path, "wb") as f:
        pickle.dump(reducer, f)

def load_umap_model(path: str) -> cuml.UMAP:

    # Load the UMAP model from the file
    loaded_reducer : cuml.UMAP

    with open(path, "rb") as f:
        loaded_reducer = pickle.load(f)

    return loaded_reducer