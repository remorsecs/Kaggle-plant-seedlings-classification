import os

import pytest
from torch.utils.data import random_split
from torchvision.transforms import transforms
from libs.dataset import PlantSeedlingsDataset, LabelNameConverter


def base_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


def test_dataset_accessible(dataset):
    try:
        print(dataset[0])
    except Exception as e:
        pytest.fail(f'Exception: {e}')


@pytest.fixture(params=['train', 'test'])
def dataset(request):
    _dataset = PlantSeedlingsDataset(
        root=os.environ['DATA_ROOT'],
        stage=request.param,
        transforms=base_transforms(),
    )
    return _dataset


def test_dataset_size(dataset):
    print(len(dataset))
    assert len(dataset) > 0


def test_split_trainset():
    _dataset = PlantSeedlingsDataset(
        root=os.environ['DATA_ROOT'],
        stage='train',
        transforms=base_transforms(),
    )
    train_size = len(_dataset) * 7 // 10
    val_size = len(_dataset) * 3 // 10
    train_set, val_set = random_split(_dataset, [train_size, val_size])
    try:
        print(train_set[0])
        print(val_set[0])
    except Exception as e:
        pytest.fail(f'Exception: {e}')


testcase_label_index = [
    ('Black-grass', 0),
    ('Charlock', 1),
    ('Cleavers', 2),
    ('Common Chickweed', 3),
    ('Common wheat', 4),
    ('Fat Hen', 5),
    ('Loose Silky-bent', 6),
    ('Maize', 7),
    ('Scentless Mayweed', 8),
    ('Shepherds Purse', 9),
    ('Small-flowered Cranesbill', 10),
    ('Sugar beet', 11),
]


@pytest.mark.parametrize('label,index', testcase_label_index)
def test_plant_seedlings_label_converter(label, index):
    converter = LabelNameConverter(root=os.environ['DATA_ROOT'])
    assert converter[label] == index
    assert converter[index] == label
