import logging

import torch

from torchvision import transforms, datasets
from .dataset import *
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image
from .autoaugment import AutoAugImageNetPolicy
import os

logger = logging.getLogger(__name__)


def get_loader(args):

    if args.dataset == "CUB":
        train_transform = transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.RandomCrop((args.img_size, args.img_size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = eval(args.dataset)(root=args.data_root, is_train=True, transform=train_transform)
        testset = eval(args.dataset)(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'AgriculturalDisease':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ColorJitter(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = AgriculturalDisease(root=args.data_root, is_train=True, transform=train_transform)
        testset = AgriculturalDisease(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'AppleLeaf9':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ColorJitter(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = AppleLeaf9(root=args.data_root, is_train=True, transform=train_transform)
        testset = AppleLeaf9(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'WheatData':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ColorJitter(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = WheatData(root=args.data_root, is_train=True, transform=train_transform)
        testset = WheatData(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'PlantPathology':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ColorJitter(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = PlantPathology(root=args.data_root, is_train=True, transform=train_transform)
        testset = PlantPathology(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'RiceLeaf':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ColorJitter(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = RiceLeaf(root=args.data_root, is_train=True, transform=train_transform)
        testset = RiceLeaf(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'PlantLeave':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ColorJitter(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = PlantLeave(root=args.data_root, is_train=True, transform=train_transform)
        testset = PlantLeave(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'PlantDoc':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ColorJitter(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = PlantDoc(root=args.data_root, is_train=True, transform=train_transform)
        testset = PlantDoc(root=args.data_root, is_train=False, transform=test_transform)


    print(trainset.__getitem__(0)[0].shape)
    print(testset.__getitem__(0)[0].shape)
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=32,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=32,
                             pin_memory=True,
                             drop_last=True) if testset is not None else None

    return train_loader, test_loader
