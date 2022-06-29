from sklearn.cluster import OPTICS, Birch, MiniBatchKMeans, DBSCAN # type: ignore
from .splitting_resources import *

def detect_split_need(dataset : ReiformICDataSet):

    min_points = min([len(dataset.get_items(c)) for c in dataset.classes()])

    def create_clf():
        # Play with this number!
        return OPTICS(min_samples=min_points//5)

    label : str = PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS

    return perform_split(dataset, label, create_clf, dataset.classes())