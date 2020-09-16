import os
from pathlib import Path

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

from libs.dataset import PlantSeedlingsDataset
from libs.model import VGG11

DATA_ROOT = os.environ['DATA_ROOT']
DEVICE = 'cpu'
# optimizer
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
# data loader
BATCH_SIZE = 16
NUM_WORKERS = 8
# training
NUM_EPOCHS = 100
SAVE_INTERNAL = 5
SAVE_ROOT = Path('params')


def build_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = PlantSeedlingsDataset(os.environ['DATA_ROOT'], transforms=transform)
    train_set_size = len(dataset) * 7 // 10
    val_set_size = len(dataset) - train_set_size
    train_set, val_set = random_split(dataset, [train_set_size, val_set_size])
    data_loader = {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
    }
    return data_loader


def update_parameters(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    device = torch.device(DEVICE)
    model = VGG11().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    data_loader = build_dataloader()

    print('Start training...')
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}:')
        # run training and validation
        for stage in ['train', 'val']:
            # fetch a batch of data from data loader
            for images, labels in data_loader[stage]:
                images, labels = images.to(device), labels.to(device)
                scores = model(images)
                loss = cross_entropy(scores, labels)

                # log
                print(f'\t{stage} loss: {loss.item():.4f}', end='\r')

                if stage == 'train':
                    update_parameters(optimizer, loss)

            if stage == 'train' and (epoch % SAVE_INTERNAL) == 0:
                torch.save(model.state_dict(), SAVE_ROOT / f'epoch-{epoch:02d}.pth')

            print()


if __name__ == '__main__':
    main()
