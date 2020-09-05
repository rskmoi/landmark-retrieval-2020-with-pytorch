from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from albumentations import CenterCrop, Compose, Normalize, RandomCrop, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset.id_to_label_3 import ID_TO_LABEL

TRAIN_ROOT = Path(__file__).resolve().parent / "../../landmark-retrieval-2020/train_over_50"
TEST_ROOT = Path(__file__).resolve().parent / "../../landmark-retrieval-2020/valid_over_40"


class _Dataset(Dataset):
    def __init__(self, mode: str = "train"):
        """
        :param mode: Uses of dataset. Must be in ["train","test"]
        """
        assert mode in ["train", "test"]
        self.mode = mode
        self.image_and_class_list = self._make_image_and_class_list()
        self.transform = self.make_transform()
        print("Created Dataset. mode: {} files: {}".format(self.mode, len(self.image_and_class_list)))

    def _make_image_and_class_list(self):
        """
        Create a list of dictionaries that contains the paths and classes of all images.
        :return: List[Dict["path": str, "class_idx": int]]
        """
        if self.mode == "train":
            root_dir = TRAIN_ROOT
        elif self.mode == "test":
            root_dir = TEST_ROOT
        else:
            raise ValueError

        image_path_list = sorted(list(root_dir.glob("*/*.jpg")))
        image_and_class_list = []
        for image_path in image_path_list:
            landmark_id = image_path.parent.name
            class_idx = ID_TO_LABEL[landmark_id] if self.mode == "train" else landmark_id
            image_and_class_list.append({"path": image_path, "class_idx": class_idx})

        return image_and_class_list

    def make_transform(self) -> Compose:
        """
        Make transform.
        TODO: Build a transform by settings in config.yml
        :return: albumentations.core.composition.Compose
        """
        if self.mode == "train":
            return Compose([Resize(256, 256),
                            RandomCrop(224, 224),
                            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ToTensorV2()])
        elif self.mode == "test":
            return Compose([Resize(256, 256),
                            CenterCrop(224, 224),
                            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ToTensorV2()])

    @property
    def num_classes(self) -> int:
        return len(ID_TO_LABEL.items())

    def __len__(self) -> int:
        return len(self.image_and_class_list)

    def __getitem__(self, idx: int):
        data = self.image_and_class_list[idx]
        path = data["path"]
        class_idx = data["class_idx"]

        image = np.array(Image.open(path))
        image = self.transform(image=image)["image"]

        return {"image": image, "label": class_idx}


class LandmarkDataset:
    def __init__(self, batch_size: int, mode: str):
        """
        :param batch_size: Batch size of loader
        :param mode: Uses of dataset. Must be in ["train","test"]
        """
        self.batch_size = batch_size
        self.num_workers = 4
        self.mode = mode
        self.dataset = _Dataset(mode=mode)

    def get_loader(self) -> DataLoader:
        loader = DataLoader(self.dataset, batch_size=self.batch_size,
                            shuffle=(self.mode == "train"), num_workers=self.num_workers)
        return loader


if __name__ == '__main__':
    dataset = LandmarkDataset(32, "train")
    print(dataset.dataset.num_classes)
    print(type(dataset.dataset.transform))
    loader = dataset.get_loader()
    for i, sample in enumerate(loader):
        print(sample["image"].shape)