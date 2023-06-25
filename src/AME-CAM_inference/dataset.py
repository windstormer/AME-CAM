import torch
from torch.utils.data import Dataset
import numpy as np
import os, glob
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle

class FeatureDataset(Dataset):
    def __init__(self, data, seg):
        self.data = data
        self.seg = seg
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        path = self.data[index]
        img_name = path.split(os.path.sep)[-1]
        # img_name = path.split("_")[-1]
        # print(img_name)
        seg_tensor = torch.zeros((1,240,240))
        for seg_path in self.seg:
            if img_name[:-3] in seg_path:
                seg_img = Image.open(seg_path)
                seg_tensor = self.transform(seg_img)
        img = Image.open(path)
        img = img.convert(mode='RGB')
        tensor = self.transform(img)

        return img_name, tensor, seg_tensor
    
    def __len__(self):
        return len(self.data)

