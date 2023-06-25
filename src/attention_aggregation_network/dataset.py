import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os

class TrainDataset(Dataset):
    def __init__(self, data, patch_size):
        self.data = data
        
        ## origin augmentation
        self.transform = transforms.Compose([
        transforms.RandomResizedCrop(patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()])
        # self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path)
        img = img.convert(mode='RGB')
        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
        if img_path.split(os.path.sep)[-2] == 'tumor':
            label = torch.ones(1)
        else:
            label = torch.zeros(1)
        return pos_1, pos_2, label
    
    def __len__(self):
        return len(self.data)

class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
        ## origin augmentation
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path)
        img = img.convert(mode='RGB')
        if img_path.split(os.path.sep)[-2] == 'tumor':
            label = torch.ones(1)
        else:
            label = torch.zeros(1)
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)
