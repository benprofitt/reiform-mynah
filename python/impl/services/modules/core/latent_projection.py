from .resources import *
from .vae_auto_net import *
from .vae_models import *
from .reiform_imageclassificationdataset import *

import torchvision.models as models # type: ignore

def read_data_get_classes(path_to_data : str) -> Tuple[torch.utils.data.DataLoader, List[str]]:
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = dataset_from_path(path_to_data)
    image_data = ImageNameDataset(data, preprocess)
    data_loader = torch.utils.data.DataLoader(image_data,
                                                batch_size=8,
                                                shuffle=True,
                                                num_workers=2)

    return data_loader, data.classes()

def pretrained_projection(data : ReiformICDataSet, model : torch.nn.Module) -> ReiformICDataSet:
    
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloader = data.get_dataloader_with_names(preprocess)

    return trained_model_projection(data, model, dataloader)

def trained_model_projection(data : ReiformICDataSet, model : nn.Module, dataloader : torch.utils.data.DataLoader) -> ReiformICDataSet:
    
    classes = data.classes()
    model = model.to(device)

    results : ReiformICDataSet = ReiformICDataSet(classes)
    
    with torch.no_grad():
        model.eval()
        for batch, label, names in dataloader:
            batch = batch.to(device)
            
            projection = model(batch)
            projection = projection.to("cpu").numpy()
            
            label = label.to("cpu").numpy()

            for idx in range(len(names)):
                file : ReiformICFile = ReiformICFile(names[idx], classes[label[idx]])
                file.add_projection(PROJECTION_LABEL, projection[idx])
                results.add_file(file)

    return results


def vae_projection(data : ReiformICDataSet, latent_size : int) -> ReiformICDataSet:

    # Find the closest power of 2 for the edge size
    sizes : Tuple[int, int, int] = data.find_max_image_size()
    max_size : int = max(sizes[0], sizes[1])

    closest_size : int = 2
    while closest_size < max_size:
        closest_size *= 2

    ReiformInfo("Size of input: {}".format(closest_size))
    # Make dataloader from dataset
    
    dataloader = data.get_dataloader(sizes[2], closest_size, CORRECTION_MODEL_BATCH_SIZE)
    proj_dataloader = data.get_dataloader(sizes[2], closest_size, CORRECTION_MODEL_BATCH_SIZE)

    ReiformInfo("Dataloader Created")
    # Make model and optimizer
    vae = VAEAutoNet(sizes[2], closest_size, latent_size)

    optimizer  = torch.optim.Adam(params=vae.parameters(), lr=0.001, weight_decay=1e-2)

    # Train the VAE
    vae, _ = train_projection_separation_vae(vae, dataloader, proj_dataloader, VAE_PROJECTION_TRAINING_EPOCHS, optimizer)

    # pass to 'pretrained projection' 
    #       - need to refactor so transform is a parameter
    vae.eval()
    results : ReiformICDataSet = trained_model_projection(data, vae.encoder, dataloader)

    # return results :-)

    return results

def inception_projection(data : ReiformICDataSet) -> ReiformICDataSet:
    inception = models.inception_v3(pretrained=True)
    results : ReiformICDataSet = pretrained_projection(data, inception)
    return results

def VGG_projection(data : ReiformICDataSet) -> ReiformICDataSet:
    vgg16 = models.vgg16(pretrained=True)
    results : ReiformICDataSet = pretrained_projection(data, vgg16)
    return results

def read_images_from_dict(files: ReiformICDataSet) -> torch.utils.data.DataLoader:
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_data = DatasetFromFilenames(files, preprocess)
    data_loader = torch.utils.data.DataLoader(image_data,
                                                batch_size=8,
                                                shuffle=True,
                                                num_workers=2)

    return data_loader

def read_projection_from_dict(files: ReiformICDataSet) -> torch.utils.data.DataLoader:

    image_data = ProjectionDataset(files)
    data_loader = torch.utils.data.DataLoader(image_data,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=2)

    return data_loader

def encode_for_2D(model : nn.Module, dataloader : torch.utils.data.DataLoader, class_list : List[str]) -> ReiformICDataSet:
    results : ReiformICDataSet = ReiformICDataSet(class_list)
    
    model.eval()

    for data, classes, filenames in dataloader:
        data = data.to(device)
        proj_2D = model(data)[0].to("cpu").detach().numpy()

        for i in range(len(filenames)):
            file : ReiformICFile = ReiformICFile(filenames[i], class_list[classes[i]])
            file.add_projection(PROJECTION_LABEL_2D, proj_2D[i])
            results.add_file(file)

    return results

def projection_2D_from_outlier_raster(inliers : ReiformICDataSet, 
                                      outliers : ReiformICDataSet):
    
    # Pair up the inliers and outliers by class
    reclassified : Dict[str, Dict[str, int]] = {}

    for clss in outliers.classes():
        reclassified[clss] = {}

        for key, _ in inliers.get_items(clss):
            reclassified[clss][key] = 0

        for key, _ in outliers.get_items(clss):
            reclassified[clss][key] = 1

    # For each class
    for clss in reclassified:
        pass
        # Train a VAE on the data - add the inlier/outlier label to the loss
        # dataloader = read_images_from_dict(reclassified[clss])
        

        # Grab the encoder and make 2D encodings of all vectors

    # Augment the input structures with the new 2D latent projections

def projection_2D_from_outlier_projection(inliers : ReiformICDataSet, outliers : ReiformICDataSet) -> Tuple[
                                                                                            ReiformICDataSet, 
                                                                                            ReiformICDataSet] :
    
    for clss in outliers.classes():

        for _, file in inliers.get_items(clss):
            file.set_was_outlier(False)

        for _, file in outliers.get_items(clss):
            file.set_was_outlier(True)

    # Pair up the inliers and outliers by class
    reclassified : ReiformICDataSet = inliers.merge(outliers)

    for clss in reclassified.classes():
        start_size : int = 0
        for _, file in inliers.get_items(clss):
            start_size = file.get_projection_size(PROJECTION_LABEL)
            break

        # Train a VAE on the data - add the inlier/outlier label to the loss
        vae = linear_VAE(start_size, 2)
        optimizer = torch.optim.Adam(params=vae.parameters(), lr=0.001, weight_decay=1e-2)
        dataloader = read_projection_from_dict(reclassified.filter_classes(clss))
        vae, loss_record = train_linear_vae(vae, dataloader, optimizer)

        # Grab the encoder and make 2D encodings of all vectors
        proj_2D : ReiformICDataSet = encode_for_2D(vae.encoder, dataloader, reclassified.classes())

        proj_out : ReiformICDataSet = proj_2D.filter_files(lambda f: f.get_was_outlier())
        proj_in : ReiformICDataSet = proj_2D.filter_files(lambda f: not f.get_was_outlier())

        outliers.merge_in(proj_out)
        inliers.merge_in(proj_in)

    # Return the new 2D projections -> {class: {filename: (<nD proj>, <2D proj>)}}
    return inliers, outliers

def projection_2D_from_outlier_projection_one_class(inliers : ReiformICDataSet, outliers : ReiformICDataSet) -> Tuple[
                                                                                        ReiformICDataSet, 
                                                                                        ReiformICDataSet] :
    
    for clss in outliers.classes():

        for _, file in inliers.get_items(clss):
            file.set_was_outlier(False)

        for _, file in outliers.get_items(clss):
            file.set_was_outlier(True)

    # Pair up the inliers and outliers by class
    reclassified : ReiformICDataSet = inliers.merge(outliers)
    
    # For each class
    for clss in reclassified.classes():
        start_size : int = 0
        for _, file in inliers.get_items(clss):
            start_size = file.get_projection_size(PROJECTION_LABEL)
            break
        break

    # Train a VAE on the data - add the inlier/outlier label to the loss
    vae = linear_VAE(start_size, 2)
    optimizer  = torch.optim.Adam(params=vae.parameters(), lr=0.001, weight_decay=1e-2)
    dataloader = read_projection_from_dict(reclassified)
    vae, loss_record = train_linear_vae(vae, dataloader, optimizer)

    # Grab the encoder and make 2D encodings of all vectors
    proj_2D : ReiformICDataSet = encode_for_2D(vae.encoder, dataloader, reclassified.classes())

    proj_out : ReiformICDataSet = proj_2D.filter_files(lambda f: f.get_was_outlier())
    proj_in : ReiformICDataSet = proj_2D.filter_files(lambda f: not f.get_was_outlier())

    outliers.merge_in(proj_out)
    inliers.merge_in(proj_in)

    # Return the new 2D projections -> {class: {filename: (<nD proj>, <2D proj>)}}
    return inliers, outliers