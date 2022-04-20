from impl.services.modules.lighting_correction.lighting_resources import *
from impl.services.modules.lighting_correction.lighting_utils import *

class DatasetForLightingDetection(torch.utils.data.Dataset):

    def __init__(self, dataset : ReiformICDataSet, transform: torchvision.transforms.Compose, all_dark: bool = False, all_light: bool = False) -> None:

        super().__init__()
        if all_dark and all_light:
            raise AssertionError("Cannot be both dark and light.")
        
        self.all_dark = all_dark
        self.all_light = all_light
        
        self.files_and_labels : List[Tuple[str, int]] = dataset._files_and_labels()

        self.transform : torchvision.transforms.Compose = transform
        self.random_balance : Dict[int, int] = {0 : 0, 1 : 0, 2 : 0}
        self.rand_limit = 128

    def __len__(self) -> int:
        return len(self.files_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        name : str = self.files_and_labels[idx][0]
        label : int = 0
        image = self.transform(Image.open(name).convert('RGB'))
        p : int = 0
        if self.all_light or self.all_dark:
            p = (1 if self.all_light else 2)
            # p = (0 if self.all_light else 1)
        else:
            p = random.randint(0,2)
            if self.random_balance[p] == self.rand_limit:
                p = (p+1) % 3
                if self.random_balance[p] == self.rand_limit:
                    p = (p+1) % 3
                    if self.random_balance[p] == self.rand_limit:
                        p = (p+1) % 3
                        for k in self.random_balance:
                            self.random_balance[k] = 0
            self.random_balance[p] = self.random_balance[p] + 1

        if p == 1:
            image = make_bright(image, rand=True)
            # label = 1
            label = 0
        elif p == 2:
            image = make_dark(image, rand=True)
            label = 1
            # label = 2

        #      image,               class
        return transforms.ToTensor()(image), label

class DatasetForLightingDetectionTensors(torch.utils.data.Dataset):

    def __init__(self, dataset : ReiformICDataSet) -> None:

        super().__init__()
        self.files_and_labels : List[Tuple[str, int]] = dataset._files_and_labels()

    def __len__(self) -> int:
        return len(self.files_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        name : str = self.files_and_labels[idx][0]
        
        label : int = 0
        p : int = 0

        if p == 1:
            label = 1
        elif p == 2:
            label = 2

        #      image,               class
        return torch.load(name), label


class DatasetForLightingCorrection(torch.utils.data.Dataset):

    def __init__(self, dataset : ReiformICDataSet, transform: torchvision.transforms.Compose) -> None:

        super().__init__()
        self.files_and_labels : List[Tuple[str, int]] = dataset._files_and_labels()

        self.transform : torchvision.transforms.Compose = transform

    def __len__(self) -> int:
        return len(self.files_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        name : str = self.files_and_labels[idx][0]
        label : int = 0
        im = Image.open(name).convert('RGB')
        p = random.randint(0,1)
        if p == 0:
            image = make_bright(im, rand=True)
        elif p == 1:
            image = make_dark(im, rand=True)
            label = 1

        #      image,                 target_im,          class
        return transforms.ToTensor()(self.transform(image)), transforms.ToTensor()(self.transform(im)), label