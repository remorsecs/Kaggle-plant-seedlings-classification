from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PlantSeedlingsDataset(Dataset):
    def __init__(self, root, stage='train', transforms=None):
        """
        :param root: path to the dataset, where contains `train`, `test` and `sample_submission.csv`
        :param stage: 'train', 'test'
        :param transforms: `torchvision.transforms`
        """
        # root/train/{Black-grass,Charlock,...}/{hash_id}.png
        # root/test/{hash_id}.png
        self._root_stage = Path(root) / stage
        self._stage = stage
        self._transforms = transforms
        self._converter = LabelNameConverter(root)
        if stage == 'train':
            self._filenames, self._labels = self._extract_filename_label()
        else:
            self._filenames = self._extract_filename()

    def _extract_filename_label(self):
        filenames = []
        labels = []

        # root_stage/{Black-grass,Charlock,...}/{hash_id}.png
        # category_folder.name: Black-grass, Charlock, ...
        for category_folder in self._root_stage.iterdir():
            filenames_in_category = [filename for filename in category_folder.iterdir()]
            filenames += filenames_in_category
            label_index = self._converter[category_folder.name]
            labels += [label_index] * len(filenames_in_category)

        return filenames, labels

    def _extract_filename(self):
        # root_stage/{hash_id}.png
        filenames = [file for file in self._root_stage.iterdir()]
        return filenames

    def __getitem__(self, index):
        with Image.open(self._filenames[index]).convert('RGB') as image:
            if self._transforms is not None:
                image = self._transforms(image)

            if self._stage == 'test':
                return image, self._filenames[index]

            return image, self._labels[index]

    def __len__(self):
        return len(self._filenames)


class LabelNameConverter(object):
    def __init__(self, root):
        root_train = Path(root) / 'train'
        self._converter = {}
        index = 0
        folders = sorted([folder for folder in root_train.iterdir()])

        for folder in folders:
            self._converter[folder.name] = index
            self._converter[index] = folder.name
            index += 1

    def __getitem__(self, index):
        return self._converter[index]


def collate_image_filename(batch: list):
    # batch = [(image_1, filename_1), (image_2, filename_2), ...]
    batch_images = []
    batch_filenames = []
    for image, filename in batch:
        batch_images.append(image)
        batch_filenames.append(filename)

    return torch.stack(batch_images), batch_filenames
