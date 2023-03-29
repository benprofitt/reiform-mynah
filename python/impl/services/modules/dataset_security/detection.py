from image_generation import *
from impl.services.modules.core.embeddings.latent_projection import EmbeddingReducer
from cuml.cluster import HDBSCAN

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
    # TODO : Rewrite to use the per-class embeddings. The whole point is to tell if the model will be tricked into swapping classes!
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
    # TODO : Rewrite to use the per-class embeddings. The whole point is to tell if the model will be tricked into swapping classes!
    origin_embeddings = original_dataset.get_embeddings_from_dataset(PROJECTION_LABEL_REDUCED_EMBEDDING)

    # Convert your data points to a cuDF dataframe
    origin_emb_df = cudf.DataFrame(origin_embeddings)

    # Set the minimum cluster size
    min_cluster_size = 10

    # Initialize the HDBSCAN algorithm with the desired parameters
    clusterer = HDBSCAN(min_samples=max(min_cluster_size, len(origin_embeddings)//10))

    # Convert the data points to a cuML matrix
    origin_emb_matrix = cuml.DataFrame.as_gpu_matrix(origin_emb_df)

    # Fit the HDBSCAN algorithm to your data
    cluster_assignments, strengths = clusterer.fit_predict(origin_emb_matrix)

    # Compare the distribution of each cluster to your known data points using a statistical test
    unique_clusters = cudf.unique(cluster_assignments)
    for i in unique_clusters:
        cluster_data = origin_emb_df[cluster_assignments == i]
        # perform a statistical test of your choice, e.g. Kolmogorov-Smirnov test


    # Check to see where embeddings of the images from each adversarial dataset land.
    
    # Each adv dataset represents a different method of creating adv images.
    # We can determine to which attack methods the dataset is most vulnerable.
    # These stats can then be in the report - and will inform how we approach the correction.

    # Fix return type!
    pass