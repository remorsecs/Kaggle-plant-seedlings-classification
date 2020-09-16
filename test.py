import os

import torch
from pandas import DataFrame
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from libs.dataset import PlantSeedlingsDataset, collate_image_filename, LabelNameConverter
from libs.model import VGG11

DEVICE = 'cuda:0'
PARAMS_PATH = 'params/exp01/epoch-49.pth'

# data loader
BATCH_SIZE = 16
NUM_WORKERS = 8


def build_data_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = PlantSeedlingsDataset(os.environ['DATA_ROOT'], stage='test', transforms=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             collate_fn=collate_image_filename)
    return data_loader


def main():
    data_loader = build_data_loader()
    device = torch.device(DEVICE)
    model = VGG11().to(device)
    model.load_state_dict(torch.load(PARAMS_PATH, map_location='cpu'))
    output = {
        'file': [],
        'species': [],
    }
    converter = LabelNameConverter(os.environ['DATA_ROOT'])
    for iteration, (images, files) in enumerate(data_loader):
        print(f'\rRun testing at iteration {iteration:02d}/{len(data_loader)}', end='')
        with torch.no_grad():
            images = images.to(device)
            scores = model(images)
            prediction_index = scores.argmax(dim=1)   # torch.tensor([2, 1, 4, ...])
            prediction_names = [converter[index.item()] for index in prediction_index]
            filenames = [file.name for file in files]

            output['file'] += filenames
            output['species'] += prediction_names

    df = DataFrame(data=output)
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
