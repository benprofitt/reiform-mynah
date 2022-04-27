from __future__ import annotations

from impl.services.modules.utils.image_utils import get_image_metadata
from .reiform_imagedataset import *

class ReiformICFile(ReiformImageFile):
    def __init__(self, name: str, label : str) -> None:
        super().__init__(name)

        self.current_class : str = label
        self.original_class : str = label

        self.confidence_vectors : List[NDArray] = []

        self.was_outlier : bool = False

    def from_json(self, body: Dict[str, Any]):
        for attrib in ["uuid", "width", "height", "channels", "current_class", "original_class", "mean", "std_dev"]:
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

    def set_was_outlier(self, value : bool = True):
        self.was_outlier = value

    def get_was_outlier(self, value: bool):
        return self.was_outlier

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
        merged_file.original_class = other.original_class

        merged_file.confidence_vectors = self.confidence_vectors + other.confidence_vectors
        merged_file.projections = self.projections.merge(other.projections)

        return merged_file

    def __deepcopy__(self, memo) -> ReiformICFile:
        
        copied : ReiformICFile = ReiformICFile(self.name, self.current_class)
        copied.original_class = self.original_class

        copied.projections = copy.deepcopy(self.projections)

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

        self.mean : List[float] = []
        self.std_dev : List[float] = []

        for c in self.class_list:
            self.files[c] = {}

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

    def add_file(self, file : ReiformICFile) -> None:
        if file.current_class not in self.class_list:
            ReiformWarning("File class {} <{}> not in dataset - file not added".format(file.current_class, str(type(file.current_class))))
            return
        self.files[file.get_class()][file.get_name()] = file

    def get_file(self, label : str, name : str) -> ReiformICFile:
            if name in self.files[label]:
                return self.files[label][name]
            else:
                raise ReiformICDataSetException("File not in dataset", "get_file")

    def remove_file(self, label : str, name : str) -> None:
        if label in self.files:
            if name in self.files[label]:
                del self.files[label][name]

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

        for _, files in self.files.items():
            for _, file in files.items():
                if random.uniform(0, 1) < ratio:
                    side_a.add_file(file)
                else:
                    side_b.add_file(file)

        return side_a, side_b

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

    def find_max_image_size(self):

        def find_max(file1 : ReiformICFile, file2 : ReiformICFile):
            new_file : ReiformICFile = ReiformICFile(file1.name, file1.current_class)

            new_file.width = max(file1.width, file2.width)
            new_file.height = max(file1.height, file2.height)
            new_file.channels = max(file1.channels, file2.channels)

            return new_file

        file = self.reduce_files(find_max)

        return file.width, file.height, file.channels

    def reduce_files(self, condition: Callable) -> ReiformICFile:

        files = self.all_files()

        file = files[0]

        for f2 in files[1:]:
            file = condition(file, f2)

        return file


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
        self = self.merge(other)

    def from_json(self, body: Dict[str, Any]):

        if len(self.class_list) == 0:
            return

        else:
            files : Dict[str, Dict[str, Dict[str, Any]]] = body["class_files"]
            for c in self.class_list:
                class_files = files[c]
                for filename, file in class_files.items():
                    new_file : ReiformICFile = ReiformICFile(filename, c)
                    new_file.from_json(file)
                    self.add_file(new_file)

            for attrib in ["mean", "std_dev"]:
                if attrib in body:
                    setattr(self, attrib, body[attrib])

    def to_json(self) -> Dict[str, Any]:
        results : Dict[str, Any] = self.__dict__

        for c in self.class_list:
            results["class_files"][c] = results["files"][c].to_json()
        

        del results["files"]

        return results

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo) -> ReiformICDataSet:
        new_ds : ReiformICDataSet = ReiformICDataSet(self.class_list)
        new_ds.files = copy.deepcopy(self.files)

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

    def get_dataloader(self, in_size: int, edge_size: int = 64, batch_size: int = 16, transformation = None) -> torch.utils.data.DataLoader:
        
        image_data = DatasetFromReiformDataset(self, in_size, edge_size, transformation)
        dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True, num_workers=workers)

        return dataloader

    def get_dataloader_with_names(self, preprocess : torchvision.transforms.Compose, batch_size: int = 16) -> torch.utils.data.DataLoader:
        image_data = ImageNameDataset(self, preprocess)
        dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True, num_workers=workers)

        return dataloader

    def mislabel(self, percent : int):
        def new_class(c : str, all_classes : List[str]):
            new_c = c
            while new_c == c:
                new_c = all_classes[random.randint(0, len(all_classes)-1)]
            return new_c

        new_dataset : ReiformICDataSet = ReiformICDataSet(self.class_list)

        for c in self.class_list:
            for _, file in self.get_items(c):
                if random.random() < percent/100.0:
                    new_c = new_class(c, self.class_list)
                    new_file : ReiformICFile = copy.deepcopy(file)
                    new_file.set_class(new_c)
                    new_dataset.add_file(new_file)
                else:
                    new_dataset.add_file(copy.deepcopy(file))
        
        return new_dataset

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

    def __len__(self) -> int:
        return len(self.files_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        name : str = self.files_and_labels[idx][0]
        label: int = self.files_and_labels[idx][1]

        if self.layer_count == 3:
            image = Image.open(name)
            image = image.convert('RGB')
            return self.transform(image), label, name
        elif self.layer_count == 1:
            return self.transform(Image.open(name).convert('L')), label, name
        else:
            raise ReiformClassMethodException("DatasetFromReiformDataset", "layer_count")


class ProjectionDataset(torch.utils.data.Dataset):

    def __init__(self, files : ReiformICDataSet) -> None:

        super().__init__()
        PROJECTION_KEY = "projection"
        self.files_projection_labels : List[Tuple[str, int, NDArray]] = files._files_labels_projections(PROJECTION_KEY)

    def __len__(self) -> int:
        return len(self.files_projection_labels)

    def __getitem__(self, idx: int) -> Tuple[str, int, NDArray]:
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

        self.transform : torchvision.transforms.Compose = transform

    def __len__(self) -> int:
        return len(self.files_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        name : str = self.files_and_labels[idx][0]
        label : int = self.files_and_labels[idx][1]
        #      image,                                           class, filename
        return self.transform(Image.open(name).convert('RGB')), label, name

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
        
        return self.transform(Image.open(name).convert('RGB')), self.image_names_and_labels[idx][1], name

def make_file(package):
    name, label = package
    new_file : ReiformICFile = ReiformICFile(name, label)

    data = get_image_metadata(name)

    new_file.width  = data["width"]
    new_file.height = data["height"]
    new_file.channels = data["channels"]

    new_file.mean = data["mean"]
    new_file.std_dev = data["std_dev"]

    return new_file

def dataset_from_path(path_to_data : str) -> ReiformICDataSet:

    image_names : List[str] = get_folder_contents(path_to_data)
    labels : List[str] = [name.split("/")[-2] for name in image_names]

    class_list = list(set(labels))
    
    dataset : ReiformICDataSet = ReiformICDataSet(class_list)

    packages = zip(image_names, labels)
    with Pool(16) as p:
        files = p.map(make_file, packages)

    for file in files:
        dataset.add_file(file)

    return dataset
