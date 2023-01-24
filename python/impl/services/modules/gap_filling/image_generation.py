import json
from detection import *
from impl.services.modules.core.reiform_imageclassificationdataset import dataset_from_path
from impl.test.image_classification.test_utils import dataset_evaluation_resnet
from impl.services.modules.core.embeddings.latent_projection import create_dataset_embedding
import torch
import torchvision

def predict_masks(dataset):
  # Load the Mask R-CNN model and set it to evaluation mode
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
  model.eval()

  # Initialize a list to store the mask predictions
  mask_predictions = []

  # Iterate through the images in the dataset
  for image, _ in dataset:
    # Add a batch dimension to the image
    image = image.unsqueeze(0)

    # Forward pass through the model to get the predictions
    predictions = model(image)

    # Append the mask predictions to the list
    mask_predictions.append(predictions['masks'])

  # Return the list of mask predictions
  return mask_predictions

def blend_images_vae(files_1 : List[ReiformICFile], files_2 : List[ReiformICFile]) -> List[ReiformICFile]:
    
    new_filenames = process_ims_vae(files_1)
    new_filenames += process_ims_vae(files_2)
    
    new_files = []
    for new_filename in new_filenames:

            # Assign the new image a unique name and the same class to make a new file
            new_files.append(ReiformICFile(new_filename, files_1[0].get_class()))

    return new_files

def blend_images(files_1 : List[ReiformICFile], files_2 : List[ReiformICFile]) -> List[ReiformICFile]:
    
    # save_images_from_files(files_1, 1)
    # save_images_from_files(files_2, 2)
    # ReiformInfo("Saved files to new location for inspection and comparison.")

    new_filenames = process_ims_vae(files_1)
    new_filenames += process_ims_vae(files_2)

    new_files = []

    pairs = []
    for file1 in files_1:
        for file2 in files_2:
            pairs.append((file1, file2))
    pairs.sort(key=lambda x: distance(x[0], x[1]))

    use_number = (len(files_1) + len(files_2)) * 3

    for file1, file2 in pairs[:use_number]:

        # new_filename = combine_images_patches([file1, file2])
        # new_filenames = new_images_grey_patch([file1, file2])
        new_filenames = swap_pixel([file1, file2])
        # new_filenames = average_image([file1, file2])
        

        for new_filename in new_filenames:

            # Assign the new image a unique name and the same class to make a new file
            new_files.append(ReiformICFile(new_filename, file1.get_class()))

    return new_files

def distance(file1 : ReiformICFile, file2 : ReiformICFile):
    p1 = file1.get_projection(GAP_PROJECTION_LABEL)
    p2 = file2.get_projection(GAP_PROJECTION_LABEL)
    if len(p1) != len(p2):
        raise ValueError("Input points must have the same number of dimensions.")
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

def test_new_images(dataset : ReiformICDataSet, test_ds : ReiformICDataSet = None):

    if test_ds is None:
        train_ds, test_ds = dataset.split(0.9)
        print("Dataset sizes: ")
        for c in train_ds.classes():
            print(len(train_ds.files[c]))
            print(len(test_ds.files[c]))
    else:
        train_ds = dataset


    train_ds, test_ds = make_gapped_dataset(train_ds, test_ds)

    for cls in train_ds.classes():
        
        plot_embeddings(train_ds, PROJECTION_LABEL_2D_PER_CLASS, [cls])
        cluster_pair_results = find_images_near_gaps(train_ds, cls)
        
        # There might be no clusters - that's okay but we should skip them
        if len(cluster_pair_results[0][0]) == 0:
            ReiformInfo("Class {} has no clusters.".format(cls))
            continue

        ReiformInfo("Class {} has clusters.".format(cls))

        augmented_dataset = train_ds.copy()
        
        for c_1, c_2 in cluster_pair_results:

            new_images = blend_images(c_1, c_2)

            for file in new_images:
                augmented_dataset.add_file(file)

        models_path = "/home/ben/Code/com.reiform.mynah/python/models"
        embedding_models_path = "{}/{}".format(models_path, EMBEDDING_MODEL_NAME)
        create_dataset_embedding(augmented_dataset, embedding_models_path)

        plot_embeddings_multi([train_ds, augmented_dataset], PROJECTION_LABEL_2D_PER_CLASS, train_ds.classes())

        # Compare a model trained on the new dataset to a model trained on the old.
        ReiformInfo("Corrected evaluation starting.")
        corrected_scores = dataset_evaluation_resnet(augmented_dataset, test_ds)
        ReiformInfo("Corrected Scores : {}".format(str(corrected_scores)))        

        ReiformInfo("Mislabeled evaluation starting.")
        raw_scores = dataset_evaluation_resnet(train_ds, test_ds)

        ReiformInfo("Raw Scores       : {}".format(str(raw_scores)))
        ReiformInfo("Corrected Scores : {}".format(str(corrected_scores)))        


if __name__ == '__main__':

    random.seed(1)

    data_path=None
    test_path=None

    if len(sys.argv) > 1:
        data_path=sys.argv[1]
    if len(sys.argv) > 2:
        test_path=sys.argv[2]

    models_path = "/home/ben/Code/com.reiform.mynah/python/models"

    dataset : ReiformICDataSet = dataset_from_path(data_path)
    embedding_models_path = "{}/{}".format(models_path, EMBEDDING_MODEL_NAME)
    create_dataset_embedding(dataset, embedding_models_path)

    if len(sys.argv) > 8:
        dataset = ReiformICDataSet([str(x) for x in range(10)])

        with open("dataset_file.json", 'r') as fh:
            obj = json.load(fh)
            obj["class_files"] = obj["files"]
            dataset.from_json(obj)

        print("Loaded dataset")

    if test_path is None:
        test_new_images(dataset)
    else:
        test_ds = dataset_from_path(test_path)
        test_new_images(dataset, test_ds)