from impl.services.modules.core.embeddings.pretrained_embedding import *
from impl.services.modules.utils.data_formatting import load_dataset

if __name__ == "__main__":

    dataset_path : str = sys.argv[1]
    dataset_name : str = sys.argv[2]
    from_path : bool = bool(int(sys.argv[3]))

    if from_path:
        dataset = dataset_from_path(dataset_path)
    else:
        dataset = load_dataset(dataset_path)

    train_embedding_for_dataset(dataset, dataset_name)