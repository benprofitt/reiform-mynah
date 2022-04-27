from .pretrained_embedding import *

if __name__ == "__main__":

    dataset_path : str = sys.argv[1]
    dataset_name : str = sys.argv[2]

    train_embedding_for_dataset(dataset_path, dataset_name)