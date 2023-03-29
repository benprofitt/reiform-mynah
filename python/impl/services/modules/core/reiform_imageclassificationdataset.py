from __future__ import annotations

from impl.services.modules.utils.image_utils import get_image_metadata
from .reiform_imagedataset import *
from torch.utils.data.sampler import WeightedRandomSampler # type: ignore

class ReiformICFile(ReiformImageFile):
    def __init__(self, name: str, label : str) -> None:
        super().__init__(name)

        self.current_class : str = label
        self.original_class : str = label

        self.confidence_vectors : List[NDArray] = []

    def from_json(self, body: Dict[str, Any]):
        for attrib in ["uuid", "width", "height", "channels", "current_class", "original_class", "mean", "std_dev"]:
            if attrib in body:
                setattr(self, attrib, body[attrib])

        if "projections" in body:
            self.projections.from_json(body["projections"])
        if "confidence_vectors" in body:
            self.confidence_vectors = [np.array(arr) for arr in body["confidence_vectors"]]

    def to_json(self) -> Dict[str, Any]:
        attr_dict : Dict[str, Any] = self.__dict__
        results : Dict[str, Any] = {}
        for k, v in attr_dict.items():
            if k not in ["projections", "confidence_vectors"]:
                results[k] = v

        results["projections"] = attr_dict["projections"].to_json()
        results["confidence_vectors"] = [v.tolist() for v in attr_dict["confidence_vectors"]]

        return results

    def serialize(self) -> Dict[str, Any]:
        results = self.to_json()

        del results["original_class"]
        del results["width"]
        del results["height"]
        del results["channels"]
        del results["name"]

        return results

    def set_class(self, label : str) -> None:
        self.current_class = label

    def get_class(self) -> str:
        return self.current_class

    def get_original_class(self) -> str:
        return self.original_class

    def count_confidence_vectors(self) -> int:
        return len(self.confidence_vectors)

    def has_confidence_vectors(self) -> bool:
        return self.count_confidence_vectors() != 0

    def add_confidence_vector(self, vec : NDArray) -> None:
        self.confidence_vectors.append(vec)

    def get_confidence_vectors(self) -> List[NDArray]:
        return self.confidence_vectors

    def clear_confidence_vectors(self) -> None:
        self.confidence_vectors = []

    def combine_projections(self, label : str, labels : List[str]):
        self.projections.combine_projections(label, labels)

    def merge(self, other: ReiformICFile) -> ReiformICFile:
        # Assuming 'other' is an updated version that is merging in - other overwrites self in conflicts
        if self.name != other.name:
            raise ReiformICFileException("Cannot merge files different names", "merge")
        merged_file : ReiformICFile = ReiformICFile(other.name, other.current_class)
        merged_file.uuid = other.uuid
        merged_file.original_class = other.original_class

        merged_file.confidence_vectors = self.confidence_vectors + other.confidence_vectors
        merged_file.projections = self.projections.merge(other.projections)

        return merged_file

    def __deepcopy__(self, memo) -> ReiformICFile:
        
        copied : ReiformICFile = ReiformICFile(self.name, self.current_class)
        copied.uuid = self.uuid
        copied.original_class = self.original_class
        copied.width = self.width
        copied.height = self.height
        copied.channels = self.channels
        copied.mean = copy.copy(self.mean)
        copied.std_dev = copy.copy(self.std_dev)

        copied.projections = copy.deepcopy(self.projections)
        copied.confidence_vectors = copy.deepcopy(self.confidence_vectors)

        return copied

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ReiformICFile):
            return False
        return (self.name == __o.name and 
                self.current_class == __o.current_class and 
                self.original_class == __o.original_class and 
                self.projections == __o.projections and 
                (np.array(self.confidence_vectors) == np.array(__o.confidence_vectors)).all())


class ReiformICDataSet(ReiformImageDataset):
    def __init__(self, classes: List[str] = []) -> None:
        super().__init__(classes)

        self.files : Dict[str, Dict[str, ReiformICFile]] = {}

        # TODO - fix to have only one map when we do a rearchitecture
        self.uuid_to_name : Dict[str, str] = {}

        self.mean : List[float] = []
        self.std_dev : List[float] = []

        self.uuid = ""

        self.max_channels = 0
        self.max_height = 0
        self.max_width = 0

        # This maps the reducer dims to the pickled reducer path on disk
        self.embedding_reducer_paths : Dict[str, Tuple[str, str]]

        for c in self.class_list:
            self.files[c] = {}
            self.uuid_to_name[c] = {}

    def add_reducer_metadata(self, proj_label : str, path : str, path_to_embedding_model : str):
        self.embedding_reducer_paths[proj_label] = (path, path_to_embedding_model)

    def get_reducer_metadata(self, proj_label : str) -> Tuple[str, str]:
        return self.embedding_reducer_paths[proj_label]

    def get_embeddings_from_dataset(self, proj_label : str) -> Tuple[List[NDArray], List[Tuple[str, str]]]:
        embeddings : List[NDArray] = []
        names : List[Tuple[str, str]] = []

        for c in self.classes():

            for name, file in self.get_items(c):
                embeddings.append(file.get_projection(proj_label))
                names.append((c, name))
        
        return embeddings, names

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ReiformICDataSet):
            return False

        if self.class_list != __o.class_list:
            return False

        return self.files == __o.files

    def empty(self) -> bool :
        return (self.file_count() == 0)

    def get_items(self, label : str):
        return self.files[label].items()

    def file_count(self) -> int:
        total = 0
        for c in self.class_list:
            total += len(self.files[c])

        return total

    def add_class(self, new_label : str) -> None:
        if new_label in self.class_list:
            ReiformMethodWarning("ReiformICDataset", "add_class", "label already exists")
            return
        self.class_list.append(new_label)
        self.files[new_label] = {}

    def add_file(self, file : ReiformICFile) -> None:
        if file.current_class not in self.class_list:
            ReiformWarning("File class {} <{}> not in dataset - file not added".format(file.current_class, str(type(file.current_class))))
            return
        self.files[file.get_class()][file.get_name()] = file
        self.uuid_to_name[file.get_class()][file.get_uuid()] = file.get_name()
        if self.max_channels < file.channels:
            self.max_channels = file.channels

    def get_file(self, label : str, name : str) -> ReiformICFile:
        label = str(label)
        
        if name in self.files[label]:
            return self.files[label][name]
        else:
            raise ReiformICDataSetException("File not in dataset", "get_file")

    def get_file_by_uuid(self, uuid : str):
        for c in self.class_list:
            if uuid in self.uuid_to_name[c]:
                return self.files[c][self.uuid_to_name[c][uuid]]
        raise ReiformICDataSetException("File not in dataset", "get_file_by_uuid")

    def get_file_by_name(self, name : str):
        for c in self.class_list:
            if name in self.files[c]:
                return self.files[c][name]
        raise ReiformICDataSetException("File not in dataset", "get_file_by_name")

    def set_file_class(self, label : str, name : str, new_label : str):
        # Removes the file from the old class and adds it back to the new one
        if new_label not in self.class_list or label not in self.class_list:
            raise ReiformDataSetException(
                        "one of classes {}, {} not in dataset".format(label, new_label),
                                        "set_file_class")
        else:
            file = self.get_file(label, name)
            file.set_class(new_label)
            self.add_file(file)
            del self.files[label][name]

    def remove_file(self, label : str, name : str) -> None:
        if label in self.files:
            if name in self.files[label]:
                del self.files[label][name]

    def minmize_projections(self) -> None:
        for c in self.class_list:
            for _, file in self.get_items(c):
                file.minmize_projection()

    def set_minus(self, other: ReiformICDataSet) -> ReiformICDataSet:
        new_ds : ReiformICDataSet = self.copy()
        if other.classes() != self.class_list:
            raise ReiformICDataSetException("Cannot subtract with different classes. \n self:{} \n other:{}".format(self.class_list, other.classes()), "merge")

        for c, files in self.files.items():
            other_files = other.files[c]
            for filename, file in files.items():
                if filename not in other_files:
                    new_ds.add_file(copy.deepcopy(file))

        return new_ds

    def split(self, ratio: float) -> Tuple[ReiformICDataSet, ReiformICDataSet]:

        side_a : ReiformICDataSet = ReiformICDataSet(classes=self.class_list)
        side_b : ReiformICDataSet = ReiformICDataSet(classes=self.class_list)

        for c in self.class_list:
            X = []
            for _, file in self.get_items(c):
                X.append(file)

            X_1, X_2 = separate_list_deque(X, ratio)

            for file in X_1:
                side_a.add_file(file)

            for file in X_2:
                side_b.add_file(file)

        return side_a, side_b

    def combine_classes(self, group : List[str]):
        for c in group:
            if c not in self.class_list:
                ReiformDataSetException("class not in dataset", "combine_classes", "ICDataset")
        main_class = group[0]
        
        for g in group[1:]:
            for name in list(self.files[g].keys()):
                self.set_file_class(g, name, main_class)
            del self.files[g]
            self.class_list.remove(g)


    def filter_classes(self, c : str) -> ReiformICDataSet:
        filtered_ds : ReiformICDataSet = ReiformICDataSet([c])
        filtered_ds.files[c] = copy.deepcopy(self.files[c])

        return filtered_ds

    def filter_files(self, condition: Callable) -> ReiformICDataSet:
        filtered_ds = self.copy()
        for c, files in self.files.items():
            for filename, file in files.items():
                if not condition(file):
                    del filtered_ds.files[c][filename]

        return filtered_ds

    def all_files(self) -> List[ReiformICFile]:
        files : List[ReiformICFile] = []

        for _, file_dict in self.files.items():
            files += list(file_dict.values())

        return files

    def find_max_image_dims(self, recalc : bool=False):

        if self.max_channels == 0 or self.max_height == 0 or self.max_width == 0 or recalc:

            def find_max(file1 : ReiformICFile, file2 : ReiformICFile):
                new_file : ReiformICFile = ReiformICFile(file1.name, file1.current_class)

                new_file.width = max(file1.width, file2.width)
                new_file.height = max(file1.height, file2.height)
                new_file.channels = max(file1.channels, file2.channels)

                return new_file

            file = self.reduce_files(find_max)

            self.max_channels = file.channels
            self.max_height = file.height
            self.max_width = file.width
            
        return self.max_width, self.max_height, self.max_channels

    def max_size(self, recalc : bool = False):
        return max(self.find_max_image_dims(recalc))

    def reduce_files(self, condition: Callable) -> ReiformICFile:
        files = self.all_files()
        if len(files) == 0:
            raise ReiformDataSetException("No files in dataset", "reduce_files", "IC")

        file = files[0]

        for f2 in files[1:]:
            file = condition(file, f2)

        return file

    def contains(self, filename : str):
        for c in self.class_list:
            if filename in self.files[c]:
                return True
        return False

    def merge(self, other : ReiformICDataSet) -> ReiformICDataSet:
        new_ds : ReiformICDataSet = self.copy()
        if other.classes() != self.class_list:
            raise ReiformICDataSetException("Cannot merge with different classes. \n self:{} \n other:{}".format(self.class_list, other.classes()), "merge")

        for c, files in self.files.items():
            other_files = other.files[c]
            for filename, file in files.items():
                if filename in other_files:
                    merged_file = file.merge(other_files[filename])
                    new_ds.add_file(merged_file)
                else:
                    new_ds.add_file(copy.deepcopy(file))

            for o_filename, o_file in other_files.items():
                if o_filename not in files:
                    new_ds.add_file(copy.deepcopy(o_file))

        return new_ds

    def intersection(self, other : ReiformICDataSet) -> ReiformICDataSet:
        new_ds : ReiformICDataSet = self.copy()
        if other.classes() != self.class_list:
            raise ReiformICDataSetException("Cannot merge with different classes. \n self:{} \n other:{}".format(self.class_list, other.classes()), "merge")

        for c, files in self.files.items():
            other_files = other.files[c]
            for filename, file in files.items():
                if filename in other_files:
                    merged_file = file.merge(other_files[filename])
                    new_ds.add_file(merged_file)
                
        return new_ds

    def merge_in(self, other : ReiformICDataSet) -> None:
        if other.classes() != self.class_list:
            raise ReiformICDataSetException("Cannot merge with different classes. \n self:{} \n other:{}".format(self.class_list, other.classes()), "merge")

        for c in other.classes():
            for filename, file in other.get_items(c):

                if self.contains(filename):
                    my_file = self.get_file_by_name(filename)
                    merged_file = my_file.merge(file)
                    self.remove_file(my_file.current_class, filename)
                    self.add_file(merged_file)
                else:
                    self.add_file(file)

    def from_json(self, body: Dict[str, Any]):

        if len(self.class_list) == 0:
            return

        else:
            files : Dict[str, Dict[str, Dict[str, Any]]] = body["class_files"]
            for c in self.class_list:
                for filename, file in files[c].items():
                    new_file : ReiformICFile = ReiformICFile(filename, c)
                    new_file.from_json(file)
                    self.add_file(new_file)

            for attrib in ["mean", "std_dev", "max_channels", "max_height", "max_width", "uuid"]:
                if attrib in body and body[attrib] is not None:
                    setattr(self, attrib, body[attrib])

    def to_json(self) -> Dict[str, Any]:
        # Use minimize projections to remove the big ones
        # self.minmize_projections()
        attr_dict : Dict[str, Any] = self.__dict__
        
        results : Dict[str, Any] = {"files" : {}}
        for k, val in attr_dict.items():
            if k != "files":
                results[k] = val

        self._files_to_json(attr_dict, results)
        
        return results

    def _serialize_files(self, attr_dict, results):
        results["class_files"] = {}
        for c in self.class_list:
            results["class_files"][c] = {}
            for name, file in attr_dict["files"][c].items():
                results["class_files"][c][name] = file.serialize()

    def _files_to_json(self, attr_dict, results):
        results["files"] = {}
        for c in self.class_list:
            results["files"][c] = {}
            for name, file in attr_dict["files"][c].items():
                results["files"][c][name] = file.to_json()

    def serialize(self) -> Dict[str, Any]:
        # Use minimize projections to remove the big ones
        # self.minmize_projections()
        attr_dict : Dict[str, Any] = self.__dict__
        results : Dict[str, Any] = {}

        extra_keys = ["files", "max_width", "max_height", "max_channels", "uuid_to_name"]
        for k, val in attr_dict.items():
            if k not in extra_keys:
                results[k] = val

        results["classes"] = results["class_list"]
        del results["class_list"]

        self._serialize_files(attr_dict, results)

        return results

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo) -> ReiformICDataSet:
        new_ds : ReiformICDataSet = ReiformICDataSet(self.class_list)
        new_ds.files = copy.deepcopy(self.files)

        new_ds.uuid = self.uuid
        new_ds.max_width = self.max_width
        new_ds.max_height = self.max_height
        new_ds.max_channels = self.max_channels
        new_ds.mean = copy.copy(self.mean)
        new_ds.std_dev = copy.copy(self.std_dev)

        return new_ds

    def combine_projections(self, label : str, labels : List[str]):
        for file in self.all_files():
            file.combine_projections(label, labels)

    def _files_and_labels(self) -> List[Tuple[str, int]]:
        files_and_labels : List[Tuple[str, int]] = []
        for i, c in enumerate(self.classes()):
            data = self.files[c]
            for filename in data:
                files_and_labels.append((filename, i))

        return files_and_labels

    def _mean_and_std_dev(self) -> None:

        def add_mean_and_std(f1 : ReiformICFile, f2 : ReiformICFile):

            nf = ReiformICFile("temp", f1.current_class)

            if len(f1.mean) != 3:
                if len(f1.mean) == 0:
                    f1.recalc_mean_and_stddev()
                if len(f1.mean) == 1:
                    f1.mean = f1.mean * 3
                    f1.std_dev = f1.std_dev * 3
                if len(f1.mean) == 4:
                    f1.mean = f1.mean[:3]
                    f1.std_dev = f1.std_dev[:3]
            if len(f2.mean) != 3:
                if len(f2.mean) == 0:
                    f2.recalc_mean_and_stddev()
                if len(f2.mean) == 1:
                    f2.mean = f2.mean * 3
                    f2.std_dev = f2.std_dev * 3
                if len(f2.mean) == 4:
                    f2.mean = f2.mean[:3]
                    f2.std_dev = f2.std_dev[:3]

            m1 = f1.mean
            m2 = f2.mean
            nf.mean = [m1[i] + m2[i] for i in range(len(m1))]

            m1 = f1.std_dev
            m2 = f2.std_dev
            nf.std_dev = [m1[i] + m2[i] for i in range(len(m1))]

            return nf

        avg_file : ReiformICFile = self.reduce_files(add_mean_and_std)

        mean_list : List[float] = avg_file.mean
        self.mean = [m/self.file_count() for m in mean_list]

        std_dev_list : List[float] = avg_file.std_dev
        self.std_dev = [m/self.file_count() for m in std_dev_list]

    def get_mean(self) -> List[float]:
        if self.mean == []:
            self._mean_and_std_dev()
        return self.mean

    def get_std_dev(self) -> List[float]:
        if self.std_dev == []:
            self._mean_and_std_dev()
        return self.std_dev

    def _files_labels_projections(self, projection_key : str) -> List[Tuple[str, int, NDArray]]:
        files_and_data : List[Tuple[str, int, NDArray]] = []
        for i, c in enumerate(self.classes()):
            data = self.files[c]
            for filename, file in data.items():
                files_and_data.append((filename, i, file.get_projection(projection_key)))

        return files_and_data

    def read_image(self, file : ReiformICFile):
        if self.max_channels == 3 or self.max_channels == 4:
            return Image.open(file.name).convert("RGB")
        else:
            return Image.open(file.name)

    def dataset_from_uuids(self, uuids : List[str]):
        
        new_dataset = ReiformICDataSet(self.class_list)
        for id in uuids:
            new_dataset.add_file(self.get_file_by_uuid(id))

        return new_dataset

    def dataset_from_names(self, names : List[str]):
        
        new_dataset = ReiformICDataSet(self.class_list)
        for name in names:
            new_dataset.add_file(self.get_file_by_name(name))

        return new_dataset

    def get_dataset(self, in_size: int, edge_size: int = 64, transformation = None) -> torch.utils.data.Dataset:
        image_data = DatasetFromReiformDataset(self, in_size, edge_size, transformation)
        return image_data

    def get_dataloader(self, in_size: int, edge_size: int = 64, batch_size: int = 16, transformation = None, shuffle = True) -> torch.utils.data.DataLoader:
        
        image_data = DatasetFromReiformDataset(self, in_size, edge_size, transformation)
        dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=shuffle, num_workers=WORKERS)
        
        return dataloader

    def get_balanced_dataloader(self, in_size : int, edge_size : int = 256, batch_size : int = 32, transformation = None) -> torch.utils.data.DataLoader:

        def class_imbalance_sampler(labels : List[int], class_count : int):
            class_counts = [0] * class_count
            for l in labels:
                class_counts[l] += 1

            class_weighting = 1. / np.array(class_counts)
            sample_weights = class_weighting[labels]
            sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)
            return sampler
        
        image_data = DatasetFromReiformDataset(self, in_size, edge_size, transformation)
        sampler = class_imbalance_sampler(image_data.get_sample_labels(), len(self.class_list))

        dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size,  
                                                 num_workers=WORKERS, sampler = sampler)

        return dataloader

    def get_dataloader_with_names(self, preprocess : torchvision.transforms.Compose, batch_size: int = 16) -> torch.utils.data.DataLoader:
        image_data = ImageNameDataset(self, preprocess)
        dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True, num_workers=WORKERS)

        return dataloader

    def mislabel(self, percent : int):
        def new_class(c : str, all_classes : List[str]):
            new_c = c
            while new_c == c:
                new_c = all_classes[random.randint(0, len(all_classes)-1)]
            return new_c

        new_dataset : ReiformICDataSet = ReiformICDataSet(self.class_list)


        count = 0
        for c in self.class_list:
            for _, file in self.get_items(c):
                if random.random() < percent/100.0:
                    count += 1
                    new_c = new_class(c, self.class_list)
                    new_file : ReiformICFile = copy.deepcopy(file)
                    new_file.set_class(new_c)
                    new_dataset.add_file(new_file)
                else:
                    new_dataset.add_file(copy.deepcopy(file))
        
        ReiformInfo("Total mislabeled: {}".format(count))
        return new_dataset, count

    def count_differences(self, other: ReiformICDataSet):
        if self.class_list != other.class_list:
            raise ReiformDataSetException("Wrong use case", "count_differences", "IC")

        count = 0
        for c in self.class_list:
            for name, file in self.get_items(c):
                if name not in other.files[c]:
                    count += 1

        return count

        
class DatasetFromReiformDataset(torch.utils.data.Dataset):

    def __init__(self, files : ReiformICDataSet, in_size: int, edge_size: int, transformation = None) -> None:
        super().__init__()
        self.files_and_labels : List[Tuple[str, int]] = files._files_and_labels()
        self.dataset : ReiformICDataSet = files
        self.layer_count = in_size

        if transformation is None:
            mean=[0.485, 0.456, 0.406] 
            std=[0.229, 0.224, 0.225]
            if in_size == 1:
                mean=[0.485]
                std=[0.229]

            self.transform : torchvision.transforms.Compose = transforms.Compose([
                transforms.Resize(edge_size),
                transforms.CenterCrop(edge_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transformation

    def get_sample_labels(self):
        return [v[1] for v in self.files_and_labels]

    def __len__(self) -> int:
        return len(self.files_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        name : str = self.files_and_labels[idx][0]
        label: int = self.files_and_labels[idx][1]

        image = self.dataset.read_image(self.dataset.get_file(self.dataset.classes()[label], name))

        return self.transform(image), label, name

class ProjectionDataset(torch.utils.data.Dataset):

    def __init__(self, files : ReiformICDataSet) -> None:

        super().__init__()
        PROJECTION_KEY = "projection"
        self.files_projection_labels : List[Tuple[str, int, NDArray]] = files._files_labels_projections(PROJECTION_KEY)

    def __len__(self) -> int:
        return len(self.files_projection_labels)

    def __getitem__(self, idx: int) -> Tuple[NDArray, int, str]:
        item : Tuple[str, int, NDArray] = self.files_projection_labels[idx]
        norm_proj : NDArray = item[2]
        norm_proj -= np.min(norm_proj)
        norm_proj /= np.max(norm_proj)
        #      projection, class,      filename
        return norm_proj,  item[1], item[0]



class DatasetFromFilenames(torch.utils.data.Dataset):

    def __init__(self, dataset : ReiformICDataSet, transform: torchvision.transforms.Compose) -> None:

        super().__init__()
        self.files_and_labels : List[Tuple[str, int]] = dataset._files_and_labels()
        self.dataset = dataset

        self.transform : torchvision.transforms.Compose = transform

    def __len__(self) -> int:
        return len(self.files_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        name : str = self.files_and_labels[idx][0]
        label : int = self.files_and_labels[idx][1]
        #      image,                                                                                               class, filename
        return self.transform(self.dataset.read_image(self.dataset.get_file(self.dataset.classes()[label], name))), label, name

class ImageNameDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: ReiformICDataSet, transform: torchvision.transforms.Compose) -> None:
        super().__init__()

        self.dataset = dataset
        self.image_names_and_labels : List[Tuple[str, int]] = dataset._files_and_labels()

        self.transform : torchvision.transforms.Compose = transform

    def get_all_classes(self) -> List[str]:
        return self.dataset.classes()

    def __len__(self) -> int:
        return len(self.image_names_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        name : str = self.image_names_and_labels[idx][0]
        label = self.image_names_and_labels[idx][1]

        return self.transform(self.dataset.read_image(self.dataset.get_file(self.dataset.classes()[label], name))), label, name

def make_file_with_RGB(package : Tuple):
    name, label = package
    new_file : ReiformICFile = ReiformICFile(name, label)

    data = get_image_metadata(name, True)

    new_file.width  = data["width"]
    new_file.height = data["height"]
    new_file.channels = data["channels"]

    new_file.mean = data["mean"]
    new_file.std_dev = data["std_dev"]

    return new_file

def make_file(package : Tuple):
    name, label = package
    new_file : ReiformICFile = ReiformICFile(name, label)

    data = get_image_metadata(name)

    new_file.width  = data["width"]
    new_file.height = data["height"]
    new_file.channels = max(data["channels"], 3)

    new_file.mean = data["mean"]
    new_file.std_dev = data["std_dev"]

    return new_file

def dataset_from_path(path_to_data : str) -> ReiformICDataSet:

    image_names : List[str] = get_folder_contents(path_to_data)
    labels : List[str] = [name.split("/")[-2] for name in image_names]

    class_list = list(set(labels))
    
    dataset : ReiformICDataSet = ReiformICDataSet(class_list)

    start = time.time()

    packages = zip(image_names, labels)

    with Pool(AVAILABLE_THREADS) as p:
        files = p.map(make_file, packages)

    for file in files:
        dataset.add_file(file)

    ReiformInfo("Time: {}".format(time.time() - start))

    return dataset
