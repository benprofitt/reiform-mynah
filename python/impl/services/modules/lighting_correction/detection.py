from .pretraining import *


def get_detection_model(dataset : ReiformICDataSet, 
                        local_model_path : str) :
    sizes = dataset.find_max_image_dims()
    max_ = max(sizes)

    edge_size = min(256, closest_power_of_2(max_))
    channels = sizes[2]

    path_to_models = get_pretrained_path_detection(local_model_path, channels, edge_size)

    model_files = glob(path_to_models + '/**/*.pt', recursive=True)
    json_files = glob(path_to_models + '/**/*.json', recursive=True)

    # Hold all new labels
    proj_labels : List[str] = []

    for m_p, j_p in zip(model_files, json_files):
        with open(j_p, "r") as fh:
            json_body = json.load(fh)
        model = load_pretrained_model(m_p, json_body, LightingDetectorSparse)

        transformation = transforms.Compose([
            transforms.Resize((edge_size, edge_size)),
            transforms.ToTensor()
        ])

        dataloader = dataset.get_dataloader_with_names(transformation)

        label = "{}_{}".format(LIGHTING_PREDICTION, m_p)
        proj_labels.append(label)
        trained_model_projection(dataset, model, dataloader, label)

    bright_files : List[str] = []
    dark_files : List[str] = []

    for c in dataset.classes():
        for name, file in dataset.get_items(c):
            all_preds : List[NDArray] = []
            for label in proj_labels:
                all_preds.append(file.get_projection(label))
                file.remove_projection(label)
            mean_pred = all_preds[0]

            for p in all_preds[1:]:
                mean_pred += p
            mean_pred /= len(all_preds)
            file.add_projection(LIGHTING_PREDICTION, mean_pred)
            index = np.argmax(mean_pred)
            if index == 1:
                bright_files.append(name)
            elif index == 2:
                dark_files.append(name)

    return dataset, {"bright" : bright_files, "dark" : dark_files}