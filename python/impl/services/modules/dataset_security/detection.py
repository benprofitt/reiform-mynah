from image_generation import *
from impl.services.modules.core.embeddings.latent_projection import EmbeddingReducer

def detect_adversarial_coverage(dataset : ReiformICDataSet):

    # Select images to have adversarial pairs created - should be ranked / values for the different methods
    candidate_dataset = get_candidates(dataset)

    # Create adversarial pairs (or more than pairs)
    methods : List[AdversarialGenerationMethod] = [AdversarialGenerationMethod.pixel_attack,
                                                   AdversarialGenerationMethod.JSM_attack]
    result_dataset_dict : Dict[AdversarialGenerationMethod, 
                               ReiformICDataSet] = generate_pairs(candidate_dataset, methods)


    # Using pre-existing embedding model, determine how many adversarial images 
    # are within the distribution of all iamges
    reducer_obj = EmbeddingReducer(embed_path, red_path, PROJECTION_LABEL_REDUCED_EMBEDDING)
    for method, adv_dataset in result_dataset_dict.items():

        red_path, embed_path = dataset.get_reducer_metadata(PROJECTION_LABEL_REDUCED_EMBEDDING)
        
        reducer_obj.perform_reduction(adv_dataset)

    distribution_results : Any = evaluate_distribution_fit(dataset, result_dataset_dict)

    pass

def find_potential_adversarial_images():
    pass

def embed_new_images(dataset : ReiformICDataSet) -> ReiformICDataSet:
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