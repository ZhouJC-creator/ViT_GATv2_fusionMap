import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
import imageio
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, ImageFolder
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg


class CUB():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]

    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class AgriculturalDisease():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'valid'
        anno_txt_file = open(os.path.join(self.root, self.mode, self.mode + '.json'))
        print(anno_txt_file.name)
        self.labels = []
        self.imgs_name = []
        with open(anno_txt_file.name) as f:
            f = json.load(f)
            for item in f:
                img_path = os.path.join(self.root, self.mode, 'images', item['image_id'])
                self.labels.append(item['disease_class'])
                self.imgs_name.append(img_path)
        # self.imgs = [scipy.misc.imread(os.path.join(self.root, 'images', img_name)) for img_name in self.imgs_name ]

    def __getitem__(self, index):
        # img_path = os.path.join(self.root, self.mode, 'images', self.imgs_name[index])
        img, target = imageio.imread(self.imgs_name[index]), self.labels[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs_name)

class AppleLeaf9():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        self.file_path = os.path.join(self.root, self.mode)
        self.len = 0

    def __getitem__(self, index):
        dataset = ImageFolder(self.file_path, transform=self.transform)
        self.len = dataset.__len__()
        img = dataset.__getitem__(index)[0]
        target = dataset.__getitem__(index)[1]
        return img, target

    def __len__(self):
        return self.len

class WheatData():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        self.file_path = os.path.join(self.root, self.mode)
        self.len = 0

    def __getitem__(self, index):
        dataset = ImageFolder(self.file_path, transform=self.transform)
        self.len = dataset.__len__()
        img = dataset.__getitem__(index)[0]
        target = dataset.__getitem__(index)[1]
        return img, target

    def __len__(self):
        return self.len

class PlantPathology():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        self.file_path = os.path.join(self.root, self.mode)
        self.len = 0

    def __getitem__(self, index):
        dataset = ImageFolder(self.file_path, transform=self.transform)
        self.len = dataset.__len__()
        img = dataset.__getitem__(index)[0]
        target = dataset.__getitem__(index)[1]
        return img, target

    def __len__(self):
        return self.len

class RiceLeaf():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        self.file_path = os.path.join(self.root, self.mode)
        self.len = 0

    def __getitem__(self, index):
        dataset = ImageFolder(self.file_path, transform=self.transform)
        self.len = dataset.__len__()
        img = dataset.__getitem__(index)[0]
        target = dataset.__getitem__(index)[1]
        return img, target

    def __len__(self):
        return self.len

class PlantLeave():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        self.file_path = os.path.join(self.root, self.mode)
        self.len = 0

    def __getitem__(self, index):
        dataset = ImageFolder(self.file_path, transform=self.transform)
        self.len = dataset.__len__()
        img = dataset.__getitem__(index)[0]
        target = dataset.__getitem__(index)[1]
        return img, target

    def __len__(self):
        return self.len

class PlantDoc():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        self.file_path = os.path.join(self.root, self.mode)
        self.len = 0

    def __getitem__(self, index):
        dataset = ImageFolder(self.file_path, transform=self.transform)
        self.len = dataset.__len__()
        img = dataset.__getitem__(index)[0]
        target = dataset.__getitem__(index)[1]
        return img, target

    def __len__(self):
        return self.len
