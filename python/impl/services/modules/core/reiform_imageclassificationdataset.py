from __future__ import annotations
from .resources import *

class Projections():
    def __init__(self) -> None:
        self.projections : Dict[str, NDArray] = {}

    def insert(self, label: str, proj : NDArray) -> None:
        self.projections[label] = proj

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

class ReiformICFile():
    def __init__(self, name : str, label : str) -> None:

        self.name : str = name
        self.current_class : str = label
        self.original_class : str = label

        self.width : int = 0
        self.height : int = 0
        self.layers : int = 3

        self.projections : Projections = Projections()
        self.confidence_vectors : List[NDArray] = []

        self.was_outlier : bool = False

    def get_projection_size(self, label: str):
        self.projections.get_size(label)

    def set_was_outlier(self, value : bool = True):
        self.was_outlier = value

    def get_was_outlier(self, value: bool):
        return self.was_outlier

    def get_name(self) -> str:
        return self.name

    def set_class(self, label : str) -> None:
        self.current_class = label

    def get_class(self) -> str:
        return self.current_class

    def get_original_class(self) -> str:
        return self.original_class

    def add_projection(self, label : str, proj : NDArray) -> None:
        self.projections.insert(label, proj)

    def get_projection(self, label : str) -> NDArray:
        return self.projections.get(label)

    def clear_projections(self) -> None:
        self.projections = Projections()

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

    def merge(self, other: ReiformICFile) -> ReiformICFile:
        # Assuming 'other' is an updated version that is merging in - other overwrites self in conflicts
        if self.name != other.name:
            raise AssertionError("Cannot merge with different names")
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


class ReiformICDataSet():
    def __init__(self, classes : List[str] = []) -> None:

        self.class_list : List[str] = classes

        self.files : Dict[str, Dict[str, ReiformICFile]] = {}
        for c in self.class_list:
            self.files[c] = {}

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ReiformICDataSet):
            return False

        if self.class_list != __o.class_list:
            return False

        return self.files == __o.files

    def classes(self) -> List[str]:
        return self.class_list

    def empty(self) -> bool :
        return (self.file_count() == 0)

    def file_count(self) -> int:
        total = 0
        for c in self.class_list:
            total += len(self.files[c])

        return total

    def add_file(self, file : ReiformICFile) -> None:
        if file.current_class not in self.class_list:
            print("Warning: File class {} <{}> not in dataset - file not added".format(file.current_class, str(type(file.current_class))))
            return
        self.files[file.get_class()][file.get_name()] = file

    def contains(self, filename : str):
        pass

    def get_file(self, label : str, name : str) -> ReiformICFile:
        if name in self.files[label]:
            return self.files[label][name]
        else:
            raise AssertionError("File not in dataset")

    def get_items(self, label : str):
        return self.files[label].items()

    def remove_file(self, label : str, name : str) -> None:
        if label in self.files:
            if name in self.files[label]:
                del self.files[label][name]

    def set_minus(self, other: ReiformICDataSet) -> None:
        raise AssertionError("need to implement ReiformICDataSet::set_minus :-)")

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
            new_file.layers = max(file1.layers, file2.layers)

            return new_file

        file = self.reduce_files(find_max)

        return file.width, file.height, file.layers

    def reduce_files(self, condition: Callable) -> ReiformICFile:

        files = self.all_files()

        file = files[0]

        for f2 in files[1:]:
            file = condition(file, f2)

        return file


    def merge(self, other : ReiformICDataSet) -> ReiformICDataSet:
        new_ds : ReiformICDataSet = self.copy()
        if other.classes() != self.class_list:
            raise AssertionError("Cannot merge with different classes. \n self:{} \n other:{}".format(self.class_list, other.classes()))

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


    def merge_in(self, other : ReiformICDataSet) -> None:
        self = self.merge(other)

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo) -> ReiformICDataSet:
        new_ds : ReiformICDataSet = ReiformICDataSet(self.class_list)
        new_ds.files = copy.deepcopy(self.files)

        return new_ds

    def _files_and_labels(self) -> List[Tuple[str, int]]:
        files_and_labels : List[Tuple[str, int]] = []
        for i, c in enumerate(self.classes()):
            data = self.files[c]
            for filename in data:
                files_and_labels.append((filename, i))

        return files_and_labels

    def _files_labels_projections(self, projection_key : str) -> List[Tuple[str, int, NDArray]]:
        files_and_data : List[Tuple[str, int, NDArray]] = []
        for i, c in enumerate(self.classes()):
            data = self.files[c]
            for filename, file in data.items():
                files_and_data.append((filename, i, file.get_projection(projection_key)))

        return files_and_data

    def get_dataloader(self, in_size: int, edge_size: int, batch_size: int = 16) -> torch.utils.data.DataLoader:
        
        image_data = DatasetFromReiformDataset(self, in_size, edge_size)
        dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True, num_workers=workers)

        return dataloader

    def get_dataloader_with_names(self, preprocess : torchvision.transforms.Compose, batch_size: int = 16) -> torch.utils.data.DataLoader:
        image_data = ImageNameDataset(self, preprocess)
        dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True, num_workers=workers)

        return dataloader

class DatasetFromReiformDataset(torch.utils.data.Dataset):

    def __init__(self, files : ReiformICDataSet, in_size: int, edge_size: int) -> None:
        super().__init__()
        self.files_and_labels : List[Tuple[str, int]] = files._files_and_labels()

        self.layer_count = in_size

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
            raise AssertionError("DatasetFromReiformDataset::layer_count not supported - submit a ticket")


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


def dataset_from_path(path_to_data : str) -> ReiformICDataSet:

    image_names : List[str] = get_folder_contents(path_to_data)
    labels : List[str] = [name.split("/")[-2] for name in image_names]

    class_list = list(set(labels))
    
    dataset : ReiformICDataSet = ReiformICDataSet(class_list)

    for name, label in zip(image_names, labels):
        new_file : ReiformICFile = ReiformICFile(name, label)
        sizes = np.array(Image.open(name)).shape

        new_file.width  = sizes[0]
        new_file.height = sizes[1]
        new_file.layers = 1 if len(sizes) < 3 else sizes[2]

        dataset.add_file(new_file)

    return dataset