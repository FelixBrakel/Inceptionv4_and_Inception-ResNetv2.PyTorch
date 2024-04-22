import logging
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model.inception_resnet_v2 import Inception_ResNetv2
from torch.optim.lr_scheduler import CosineAnnealingLR


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_distrib(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data/cifar100',
        train=True,
        transform=train_transform,
        download=True
    )
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )

    # create model and move it to GPU with id rank
    model = Inception_ResNetv2(classes=100).to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.025)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    logger = logging.getLogger()
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    for epoch in range(100):
        sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(rank)
            labels = labels.to(rank)

            outputs = model(images)
            loss = criterion(outputs, labels)
            if i % 25 == 0:
                logger.log(
                    logging.INFO,
                    f"Epoch {epoch}-{i}, Train loss: {loss:.5f}, Learning rate: {scheduler.get_last_lr()}",
                )

            loss.backward()
            optimizer.step()

        scheduler.step()

    if rank == 0:
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data/cifar100',
            train=False,
            transform=valid_transform
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(rank)
                labels = labels.to(rank)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    cleanup()


def run(run_fn, world_size):
    mp.spawn(run_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def main():
    print(f"Cuda available? {torch.cuda.is_available()}")
    n_gpus = torch.cuda.device_count()
    print(f"running with {n_gpus} gpus")
    run(train_distrib, 1)


if __name__ == '__main__':
    main()
