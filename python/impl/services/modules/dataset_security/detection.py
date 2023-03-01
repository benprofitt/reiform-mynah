from image_generation import *

def detect_adversarial_coverage(dataset : ReiformICDataSet):

    # Select images to have adversarial pairs created - should be ranked / values for the different methods
    candidate_dataset = get_candidates(dataset)

    # Create adversarial pairs (or more than pairs)
    result_dataset_dict : Dict[AdversarialGenerationMethod, ReiformICDataSet] = generate_pairs(candidate_dataset)


    # Using pre-existing embedding model, determine how many adversarial images 
    # are within the distribution of all iamges

    distribution_results : Any = evaluate_distribution_fit(dataset, )

    pass

def find_potential_adversarial_images():
    pass

def embbed_new_images(dataset : ReiformICDataSet) -> ReiformICDataSet:
    pass

def evaluate_distribution_fit(original_dataset : ReiformICDataSet, 
                              adversarial_datasets : Dict[AdversarialGenerationMethod, 
                                                          ReiformICDataSet]) -> Tuple[float,
                                                                               ReiformICDataSet,
                                                                               ReiformICDataSet]:
    # Get the distribution of the original image set embeddings

    # Check to see where embeddings of the images from each adversarial dataset land.
    
    # Each adv dataset represents a different method of creating adv images.
    # We can determine to which attack methods the dataset is most vulnerable.
    # These stats can then be in the report - and will inform how we approach the correction.

    # Fix return type!
    pass