"""Data handler file.
Used to download all garfield strips and put it in a dataset.
Call prepare_data_loader(), which returns train/validation/test DataLoaders.
"""

import requests
import shutil
import os
from datetime import date, timedelta

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt

def daterange(start_date, end_date):
    """Generator for a range of dates"""
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def download_all_strips(root='data/all_images/', force_download=False):
    if not os.path.exists(root):
        os.makedirs(root)
    elif not force_download:
        return #assume already downloaded

    pic_url = 'http://picayune.uclick.com/comics/ga/%Y/ga%y%m%d.gif'
    save_path_base = f'{root}%y%m%d.jpg'

    start_date = date(1978, 6, 19)
    end_date = date.today()
    for single_date in daterange(start_date, end_date):
        if single_date.weekday() == 6:
            continue
        download_url = single_date.strftime(pic_url)
        save_path = single_date.strftime(save_path_base)

        img_data = requests.get(download_url).content
        with open(save_path, 'wb') as handler:
            handler.write(img_data)

def make_archive(root='data/all_images/'):
    #Note: unused
    if os.path.exists(root):
        shutil.make_archive(f'{root}', 'zip', root)
        shutil.rmtree(root)

class GarfieldDataset(Dataset):
    def __init__(self, root='data/all_images/'):
        self.data = []
        for img in os.listdir(root):
            self.data.append([os.path.join(root, img), 1])
        self.img_dim = (175, 200)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        pic = plt.imread(img_path)[:,:,:3]

        # Resize the image from (600,175) to (600,176), this is because
        # 176 and 200 (width of 1 panel) is divisible by 8.
        pic = cv2.resize(pic, (600,176))
        pic_tensor = torch.from_numpy(pic)
        split = pic_tensor.chunk(3, dim=1)
        splitted_tensor = torch.stack(split)
        return splitted_tensor

def prepare_data_loader(root='data/all_images/', batch_size=128, num_workers=4, force_download=False):
    download_all_strips(force_download=force_download)
    dataset = GarfieldDataset()

    # Only use 10% for training and validation, this is due to memory constraints
    # Could be increased to potentionally improve performance/reduce overfitting
    train_size = int(0.15 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False)

    return train_loader, val_loader, test_loader
