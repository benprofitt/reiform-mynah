from sklearn.cluster import OPTICS # type: ignore
from .splitting_resources import *

def split_dataset(dataset : ReiformICDataSet, classes_to_cluster : List[str]) -> Tuple[ReiformICDataSet, Dict[str, Dict[str, List[str]]]]:

    min_points = min([len(dataset.get_items(c)) for c in classes_to_cluster])

    def create_clf():
        return OPTICS(min_samples=min_points//4, min_cluster_size=min_points//5)

    label : str = PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS

    potential_cluster_count, potential_clusters = perform_split(dataset, label, create_clf, classes_to_cluster)

    new_label_to_files : Dict[str, Dict[str, List[str]]] = {}

    # relabel classes in files, and add class names to dataset as needed
    for c, value in potential_cluster_count.items():
        new_label_to_files[c] = {}

        # The first split stays as the original class
        new_split = potential_clusters[c][1:]

        new_class_names : List[str] = []

        for i in range(value-1):

            new_class_name = "{}_split_{}".format(c, i+2)

            while new_class_name in dataset.classes():
                new_class_name += "_new"
            new_class_names.append(new_class_name)
            new_label_to_files[c][new_class_name] = []

        for i, name in enumerate(new_class_names):
            dataset.add_class(name)

            for file_name in new_split[i]:
                dataset.set_file_class(c, file_name, name)
                new_label_to_files[c][name].append(file_name)
    
    return dataset, new_label_to_files