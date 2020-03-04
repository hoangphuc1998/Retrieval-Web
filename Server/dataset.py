from torch.utils.data import Dataset
import torch
import os
import json
import pickle
from PIL import Image


class ImageDataset(Dataset):
    '''
    Load all images from a folder
    '''
    # TODO: Add dataset for LSC2020 (this only apply to COCO)

    def __init__(self, image_folder, transform):
        self.image_folder = image_folder
        self.transform = transform
        self.filelist = os.listdir(self.image_folder)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        name = self.filelist[idx]
        image = Image.open(os.path.join(
            self.image_folder, name)).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor, name


class FeatureDataset(Dataset):
    '''
    Load all features after extracted from a folder
    '''
    # TODO: Add dataset for LSC2020 (this only apply to COCO)

    def __init__(self, folder, device):
        self.folder = folder
        self.filelist = os.listdir(folder)
        self.device = device

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        image_file = os.path.splitext(filename)[0] + '.jpg'
        path = os.path.join(self.folder, filename)
        feature = torch.load(path).to(self.device).detach()
        return feature, image_file

    def __len__(self):
        return len(self.filelist)