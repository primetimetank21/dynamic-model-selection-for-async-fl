from pathlib import Path
import warnings
import opendatasets as od
from typing import Optional, Callable, Tuple, Dict
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset


## Define custom CobaDataset class
class COBA(VisionDataset):
    """
    `COBA <https://www.kaggle.com/datasets/earltankardjr/coba-iobt-dataset> Dataset`
    Args:
        root (string): Root directory of dataset where ``COBA/raw/data.pt``
            and  ``COBA/raw/targets.pt`` exist.
        train (bool, optional): If `True`, creates dataset from ``training.pt``,
            otherwise from ``test.pt``. -- This currently doesn't do anything!
        download (bool, optional): If `True`, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources: list = [
        "https://www.kaggle.com/datasets/earltankardjr/coba-iobt-dataset"
    ]  # link(s) to coba dataset
    training_file: str = "training.pt"
    test_file: str = "test.pt"
    # training_file:str = "training.pt"
    # test_file:str = "test.pt"
    classes: list = [
        "0 - airplane",
        "1 - ambulance",
        "2 - briefcase",
        "3 - cannon",
        "4 - car",
        "5 - civilian",
        "6 - dagger",
        "7 - dog",
        "8 - handgun",
        "9 - missilerocket",
        "10 - rifle",
        "11 - soldier",
        "12 - tank",
        "13 - truck",
    ]

    @property
    def train_labels(self) -> torch.Tensor:
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self) -> torch.Tensor:
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self) -> torch.Tensor:
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self) -> torch.Tensor:
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        # if self.train:
        #     data_file = self.training_file
        # else:
        #     data_file = self.test_file

        self.data, self.targets = self._load_data()

    def download(self) -> None:
        if self._check_exists():
            return

        # Make Raw directory
        Path.mkdir(self.raw_folder, exist_ok=True, parents=True)

        # Download dataset file
        coba_dataset_url: str = self.resources[0]
        od.download(coba_dataset_url)

        # Move dataset
        coba_iobt_npy_file: str = "iobt_128_128.npy"

        downloaded_coba_dataset_path = Path.cwd().joinpath(
            "coba-iobt-dataset", coba_iobt_npy_file
        )

        if not downloaded_coba_dataset_path.exists():
            raise FileNotFoundError("Failed to download and locate COBA dataset")

        new_coba_dataset_path = Path.cwd().joinpath(self.raw_folder, coba_iobt_npy_file)
        downloaded_coba_dataset_path.rename(new_coba_dataset_path)

        # Format dataset
        with open(new_coba_dataset_path, "rb") as f:
            data = torch.tensor(np.load(f), dtype=torch.float32)
            targets = torch.tensor(np.load(f, allow_pickle=True), dtype=torch.int64)

        # Save dataset
        torch.save(data, self.raw_folder.joinpath("data.pt"))
        torch.save(targets, self.raw_folder.joinpath("targets.pt"))

        # Delete unnecessary files/directories
        Path.rmdir(Path.cwd().joinpath("coba-iobt-dataset"))
        Path.unlink(Path.cwd().joinpath(self.raw_folder, coba_iobt_npy_file))

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file: str = "data.pt"
        data = torch.load(Path.cwd().joinpath(self.raw_folder, image_file))

        label_file: str = "targets.pt"
        targets = torch.load(Path.cwd().joinpath(self.raw_folder, label_file))

        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, label

    @property
    def raw_folder(self) -> Path:
        return Path(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> Path:
        return Path(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[int, str]:
        # class_dict:dict[int,str] = {int(label.replace(" ","").split("-")[0]): label.replace(" ","").split("-")[1] for label in self.classes} #one liner
        class_dict: Dict[int, str] = {}
        for label in self.classes:
            encoded_val, name = label.replace(" ", "").split("-")
            class_dict[int(encoded_val)] = name
        return class_dict

    def _check_exists(self) -> bool:
        return Path.exists(Path(self.raw_folder, "data.pt")) and Path.exists(
            Path(self.raw_folder, "targets.pt")
        )
        # return (Path.exists(Path.joinpath(self.raw_folder, self.training_file)) and Path.exists(Path.joinpath(self.raw_folder, self.test_file)))
