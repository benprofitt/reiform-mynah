from impl.services.modules.core.resources import *
from impl.services.modules.core.vae_auto_net import *
from impl.services.modules.core.vae_models import *
from impl.services.modules.core.reiform_imageclassificationdataset import *

from .pretrained_embedding import *


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

def get_dataset_embedding(dataset : ReiformICDataSet, path_to_embeddings : str):
    
    # Get all model files under this ^ path, 
    # process dataset, and add the projections to 
    # the dataset. Then combine all of the projections, 
    # and use umap_red to reduce to fewer dimensions. (for outliers)
    # Finally, reduce again to 2D (for reporting)

    start = time.time()

    # Load a resnet from the model zoo
    model_ = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
    model_ = model_._model

    return_nodes = {
        "features": "embedding"
    }
    model_ = create_feature_extractor(model_, return_nodes=return_nodes)
    model_.to(device)
    model_.eval()
    pt_dl = dataset.get_dataloader(3, batch_size=96, edge_size=224)

    embeddings : List[NDArray] = []
    file_names : List[str] = []

    for images, labels, name_batch in pt_dl:
        images = images.to(device)

        pred = model_(images)["embedding"]
        x = nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        x = torch.flatten(x, 1)

        file_names += name_batch
        for v in x.to("cpu").detach().numpy():
            embeddings.append(v)

    ReiformInfo("Time: {}".format(time.time() - start))
    start = time.time()

    for i, name in enumerate(file_names):
        file = dataset.get_file_by_uuid(name)
        file.add_projection(PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION, embeddings[i])

    ReiformInfo("Time: {}".format(time.time() - start))
    start = time.time()

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
    
    ReiformInfo("Time: {}".format(time.time() - start))
    start = time.time()

    for c in dataset.classes():
        # Per-class embedding reduction -> used for class splitting
        umap_red = umap.UMAP(n_components=EMBEDDING_DIM_SIZE, n_jobs=8)
        reduced_embeddings = umap_red.fit_transform(embeddings_by_class[c])
        for i, name in enumerate(names_by_class[c]):
            dataset.get_file(c, name).add_projection(PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS, reduced_embeddings[i])

        umap_red = umap.UMAP(n_jobs=8)
        reduced_embeddings = umap_red.fit_transform(embeddings_by_class[c])
        for i, name in enumerate(names_by_class[c]):
            dataset.get_file(c, name).add_projection(PROJECTION_LABEL_2D_PER_CLASS, reduced_embeddings[i])

    ReiformInfo("Time: {}".format(time.time() - start))
    start = time.time()

    # Entire dataset embedding reduction -> used for outlier detection
    umap_red = umap.UMAP(n_components=EMBEDDING_DIM_SIZE, n_jobs=8)
    reduced_embeddings = umap_red.fit_transform(embeddings)

    for i, file in enumerate(names):
        dataset.get_file(file[0], file[1]).add_projection(PROJECTION_LABEL_REDUCED_EMBEDDING, reduced_embeddings[i])

    # 2D projections -> Used to show user what's up with these embeddings
    umap_red = umap.UMAP(n_jobs=8)
    reduced_embeddings = umap_red.fit_transform(embeddings)

    for i, file in enumerate(names):
        dataset.get_file(file[0], file[1]).add_projection(PROJECTION_LABEL_2D, reduced_embeddings[i])

    ReiformInfo("Time: {}".format(time.time() - start))
    start = time.time()