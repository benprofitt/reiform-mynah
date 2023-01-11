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

def blend_images(files_1 : List[ReiformICFile], files_2 : List[ReiformICFile]) -> List[ReiformICFile]:
    
    new_files = []

    pairs = []
    for file1 in files_1:
        for file2 in files_2:
            pairs.append((file1, file2))
    pairs.sort(key=lambda x: distance(x[0], x[1]))

    use_number = len(files_1) + len(files_2)

    for file1, file2 in pairs[:use_number]:
        # load the two files
        im1 = load_image(file1.get_name())
        im2 = load_image(file2.get_name())

        # join them in some way? to make new image
        im_new = (im1 + im2)//2
        
        # Generate a random UUID for the name - uuid4 gives more privacy that uuid1
        uuid4 = uuid.uuid4()
        new_filename = "new_images/{}.png".format(uuid4)

        # Assign the new image a unique name and the same class to make a new file
        save_image(im_new, new_filename)
        new_files.append(ReiformICFile(new_filename, file1.get_class()))

    print(len(new_files))

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
    else:
        train_ds = dataset


    for cls in train_ds.classes():
        
        plot_embeddings(train_ds, PROJECTION_LABEL_2D_PER_CLASS, [cls])
        cluster_pair_results = find_images_near_gaps(train_ds, cls)
        
        # There might be no clusters - that's okay but we should skip them
        if len(cluster_pair_results[0][0]) == 0:
            ReiformInfo("Class {} has no clusters.".format(cls))
            continue

        augmented_dataset = train_ds.copy()
        
        for c_1, c_2 in cluster_pair_results:

            new_images = blend_images(c_1, c_2)

            for file in new_images:
                augmented_dataset.add_file(file)

        # Compare a model trained on the new dataset to a model trained on the old.
        ReiformInfo("Mislabeled evaluation starting.")
        raw_scores = dataset_evaluation_resnet(train_ds, test_ds)
        ReiformInfo("Corrected evaluation starting.")
        corrected_scores = dataset_evaluation_resnet(augmented_dataset, test_ds)

        ReiformInfo("Raw Scores       : {}".format(str(raw_scores)))
        ReiformInfo("Corrected Scores : {}".format(str(corrected_scores)))        


if __name__ == '__main__':

    data_path=None
    test_path=None

    if len(sys.argv) > 1:
        data_path=sys.argv[1]
    if len(sys.argv) > 2:
        test_path=sys.argv[2]

    # models_path = "/home/ben/Code/com.reiform.mynah/python/models"

    # dataset : ReiformICDataSet = dataset_from_path(data_path)
    # embedding_models_path = "{}/{}".format(models_path, EMBEDDING_MODEL_NAME)
    # create_dataset_embedding(dataset, embedding_models_path)

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