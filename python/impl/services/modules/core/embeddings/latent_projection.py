from impl.services.modules.core.resources import *
from impl.services.modules.core.vae_auto_net import *
from impl.services.modules.core.vae_models import *
from impl.services.modules.core.reiform_imageclassificationdataset import *

from .pretrained_embedding import *

def pretrained_projection(data : ReiformICDataSet, model : torch.nn.Module, 
                          preprocess : transforms.Compose, label : str) -> ReiformICDataSet:
    
    dataloader = data.get_dataloader_with_names(preprocess)

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
                file : ReiformICFile = data.get_file(label[idx], names[idx])
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


def get_dataset_embedding(dataset : ReiformICDataSet, path_to_embeddings : str):
    sizes, max_ = max_sizes(dataset)
    channels = sizes[2]
    closest_size = closest_power_of_2(max_)

    path_to_models = get_embedding_path(channels, closest_size, local_path=path_to_embeddings)

    # TODO: Get all model files under this ^ path, 
    # process dataset, and add the projections to 
    # the dataset. Then combine all of the projections, 
    # and use PCA to reduce to fewer dimensions. (for outliers)
    # Finally, reduce again to 2D (for reporting)

    model_files = glob(path_to_models + '/**/*.pt', recursive=True)
    json_files = glob(path_to_models + '/**/*.json', recursive=True)

    for model_path, json_path in zip(model_files, json_files):

        with open(json_path, 'r') as fh:
            json_body = json.load(fh)

            edge_size : int = json_body[SIZE]
            resize : str = json_body[RESIZE]
            resize_str = "transforms.Resize" + ("({})".format(edge_size) if resize == "min_size" else "(({}, {}))".format(edge_size, edge_size))

            transformation = transforms.Compose([
                eval(resize_str),
                transforms.RandomCrop(edge_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=json_body[MEAN], std=json_body[STD])
            ])

        encoder = load_embedding_model(model_path, json_body)

        dataset = pretrained_projection(dataset, encoder, transformation, json_path)

    dataset.combine_projections(PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION, json_files)

    embeddings : List[NDArray] = []
    names : List[Tuple[str, str]] = []
    for c in dataset.classes():
        for name, file in dataset.get_items(c):

            embeddings.append(file.get_projection(PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION))
            names.append((c, name))
    
    # Embedding for outlier detection
    pca = PCA(n_components="mle")
    reduced_embeddings = pca.fit_transform(embeddings)

    for i, (c, name) in enumerate(names):
        dataset.get_file(c, name).add_projection(PROJECTION_LABEL_REDUCED_EMBEDDING, reduced_embeddings[i])

    # 2D projections
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    for i, (c, name) in enumerate(names):
        dataset.get_file(c, name).add_projection(PROJECTION_LABEL_2D, reduced_embeddings[i])