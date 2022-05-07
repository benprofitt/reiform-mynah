from .correction_models import *
from impl.services.modules.lighting_correction.pretraining import get_pretrained_path_lighting, load_pretrained_model

class AdjustedLighting(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(original, correction):
        delta = original - correction
        delta = torch.mul(delta,0.75)
        return original - delta


def run_correction_model(local_model_path : str, dataset : ReiformICDataSet):
    model, dataloader = load_correction_model(local_model_path, dataset)

    model.to(device)
    model.eval()

    apply_func = AdjustedLighting()

    # For each image
    for images, labels, names in dataloader:

        images = images.to(device)

        corrected = model(images).to('cpu').detach()
        images = images.to("cpu")

        for i , name in names:
            # Run the model on image
            corr_im = corrected[i] # Pytorch tensor
            c = dataset.classes()[labels[i]]
            file = dataset.get_file(c, name)

            # Get the corrected version, reshape to original image shape
            trans = transforms.Compose([
                transforms.Resize((file.height, file.width))
            ])
            corr_im = trans(corr_im)
            original = images[i]
            
            # For each pixel in the original image, change it 75% to the new value in the correct version
            adjusted_image = apply_func(original, corr_im)

            # Save the new version
            file.save_image(transforms.ToPILImage()(adjusted_image))


def load_correction_model(local_model_path : str, dataset : ReiformICDataSet):

    sizes = dataset.find_max_image_dims()
    edge_size = max(64, min(1024, closest_power_of_2(max(sizes[:2]))*2))
    edge_size = (edge_size if edge_size in LIGHTING_CORRECTION_EDGE_SIZES else edge_size*2)
    channels = sizes[2]

    path_to_models = get_pretrained_path_lighting(local_model_path, channels, edge_size)

    model_file = glob(path_to_models + '/**/*.pt', recursive=True)[0]
    json_file = glob(path_to_models + '/**/*.json', recursive=True)[0]

    with open(json_file, "r") as fh:
        json_body = json.load(fh)
        model = load_pretrained_model(model_file, json_body, LightingCorrectionNet)

        transformation = transforms.Compose([
            transforms.Resize((edge_size, edge_size)),
            transforms.ToTensor()
        ])

        dataloader = dataset.get_dataloader_with_names(transformation)

        return model, dataloader