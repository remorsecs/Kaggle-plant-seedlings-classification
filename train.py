import os
from pathlib import Path

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

from libs.dataset import PlantSeedlingsDataset
from libs.model import VGG11
from libs.visualization import LossPresenter, LossVisualizer

DATA_ROOT = os.environ['DATA_ROOT']
DEVICE = 'cpu'
# DEVICE = 'cuda'

# optimizer
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
# data loader
BATCH_SIZE = 32
NUM_WORKERS = 8
# training
NUM_EPOCHS = 100
SAVE_INTERNAL = 5
EXP_NAME = 'exp01'
SAVE_ROOT = Path('params') / EXP_NAME
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT = Path('logs') / EXP_NAME
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def build_data_loader():
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(90),
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
    data_loader = build_data_loader()
    loss_presenter = LossPresenter()

    print('Start training...')
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}:')

        # run training and validation
        train_loss = 0.
        val_loss = 0.
        for stage in ['train', 'val']:
            num_corrects = 0.
            # fetch a batch of data from data loader
            for iteration, (images, labels) in enumerate(data_loader[stage]):
                with torch.set_grad_enabled(stage == 'train'):
                    images, labels = images.to(device), labels.to(device)
                    scores = model(images)
                    loss = cross_entropy(scores, labels)

                    # log
                    print(f'\r\t{stage} loss: {loss.item():.4f} at iteration {iteration:02d}/{len(data_loader[stage])}',
                          end='', flush=True)
                    num_corrects += sum(scores.argmax(dim=1) == labels)

                    if stage == 'train':
                        update_parameters(optimizer, loss)
                        train_loss += loss.item()
                    else:
                        val_loss += loss.item()

            if stage == 'train' and ((epoch + 1) % SAVE_INTERNAL) == 0:
                torch.save(model.state_dict(), SAVE_ROOT / f'epoch-{epoch:02d}.pth')

            accuracy = num_corrects / len(data_loader[stage].dataset)
            print(f'\taccuracy: {accuracy:.4f}')

        train_loss /= len(data_loader['train'])
        val_loss /= len(data_loader['val'])
        loss_presenter.update(epoch, train_loss, val_loss)

    visualizer = LossVisualizer(loss_presenter)
    visualizer.save(LOG_ROOT / 'loss.png')


if __name__ == '__main__':
    main()
