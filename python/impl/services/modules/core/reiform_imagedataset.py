from __future__ import annotations

from impl.services.modules.utils.image_utils import get_image_metadata
from .resources import *

class Projections():
    def __init__(self) -> None:
        self.projections : Dict[str, NDArray] = {}

    def from_json(self, projs : Dict[str, List[float]]) -> None:
        for name, arr in projs.items():
            self.projections[name] = np.array(arr)

    def to_json(self) -> Dict[str, List[float]]:
        results : Dict[str, List[float]] = {}
        for name, arr in self.projections.items():
            results[name] = arr.tolist()

        return results

    def insert(self, label: str, proj : NDArray) -> None:
        self.projections[label] = proj

    def remove(self, label : str) -> None:
        if label in self.projections:
            del self.projections[label]

    def get(self, label: str) -> NDArray:
        return self.projections[label]

    def get_size(self, label: str) -> int:
        return self.get(label).size

    def merge(self, other: Projections) -> Projections:
        new_proj : Projections = Projections()

        for key, val in self.projections.items():
            new_proj.insert(key, val)

        for key, val in other.projections.items():
            new_proj.insert(key, val)

        return new_proj

    def combine_projections(self, label : str, labels : List[str]):
        arr = np.array([])
        for l in labels:
            arr = np.concatenate((arr, self.projections[l]))
        self.insert(label, arr)

    def __deepcopy__(self, memo) -> Projections:
        copied : Projections = Projections()
        copied.projections = copy.deepcopy(self.projections)
        return copied

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Projections):
            return False
        if len(self.projections) != len(__o.projections):
            return False

        for k, arr in self.projections.items():
            if k in __o.projections:
                if not (arr == __o.projections[k]).all():
                    return False

            else:
                return False

        return True

class ReiformImageDataset():
    def __init__(self, classes : List[str] = []) -> None:
        self.class_list : List[str] = classes

    def classes(self) -> List[str]:
        return self.class_list

    def contains(self, filename : str):
        pass


class ReiformImageFile():
    def __init__(self, name : str) -> None:

        self.uuid : str = ""
        self.name : str = name

        self.width : int = 0
        self.height : int = 0
        self.channels : int = 3

        self.mean : List[float] = []
        self.std_dev : List[float] = []

        self.projections : Projections = Projections()

    def get_projection_size(self, label: str):
        self.projections.get_size(label)

    def get_uuid(self) -> str:
        return self.uuid

    def get_name(self) -> str:
        return self.name

    def add_projection(self, label : str, proj : NDArray) -> None:
        self.projections.insert(label, proj)

    def get_projection(self, label : str) -> NDArray:
        return self.projections.get(label)

    def remove_projection(self, label : str) -> None:
        self.projections.remove(label)

    def minmize_projection(self) -> None:
        projs = self.projections.projections
        if PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION in projs:
            projs[PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION] = np.array([0.0, 1.0, 2.0, 3.0])

    def clear_projections(self) -> None:
        self.projections = Projections()

    def recalc_mean_and_stddev(self) -> None:
        d = get_image_metadata(self.name, len(self.mean) == 3)
        self.mean = d["mean"]
        self.std_dev = d["std_dev"]

    def save_image(self, image : Image.Image) -> None:
        ReiformWarning("Overwriting file at {}".format(self.name))
        im_types = {1 : "L", 3 : "RGB", 4 : "RGBA"}
        if self.channels in im_types:
            im_type = im_types[self.channels]
            image.convert(im_type).save(self.name)
        else:
            image.save(self.name)

    def move_image(self, path : str, delete : bool = False) -> None:
    
        new_location = "{}/{}".format(path, self.name.split("/")[-1])
        Image.open(self.name).save(new_location)

        if delete:
            ReiformWarning("Deleting file at {}".format(self.name))
            os.remove(self.name)

        self.name = new_location


class ReiformImageEntity():
    def __init__(self, label : str) -> None:
        self.current_class : str = label
        self.original_class : str = label

        self.confidence_vectors : List[NDArray] = []
        self.projections : Projections = Projections()

    def from_json(self, body : Dict[str, Any]):
        for attrib in ["current_class", "original_class"]:
            if attrib in body:
                setattr(self, attrib, body[attrib])

        if "projections" in body:
            self.projections.from_json(body["projections"])
        if "confidence_vectors" in body:
            self.confidence_vectors = [np.array(arr) for arr in body["confidence_vectors"]]

    def to_json(self) -> Dict[str, Any]:
        results : Dict[str, Any] = self.__dict__

        results["projections"] = results["projections"].to_json()
        results["confidence_vectors"] = [v.tolist() for v in results["confidence_vectors"]]

        return results
