from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets
import numpy as np
import pandas as pd
from PIL import Image
import os, os.path
import ntpath
import scipy.misc
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class CNNDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.dataset = CNNData(self.data_dir)

        super(CNNDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CNNData(Dataset):
    def __init__(self, img_path):
        """
        Args:
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.classDict = {}
        # Transforms
        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.to_tensor = transforms.ToTensor()
        # Calculate len
        self.data_len = len([name for name in os.listdir(img_path) if os.path.isfile(img_path + name)])
        self.image_arr = [img_path + name for name in os.listdir(img_path) if os.path.isfile(img_path + name)]

    def checkClass(self,pathAndImg):
        name = ntpath.basename(pathAndImg)
        name = os.path.splitext(name)[0]
        name = ''.join([i for i in name if not i.isdigit()])
        if name not in self.classDict:
            self.classDict[name] = len(self.classDict)
        return self.classDict[name]

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        target = self.checkClass(single_image_name)
        img_as_img = Image.open(single_image_name).convert('L')
        # If there is an operation
        if self.trsfm:
            img_as_tensor = self.trsfm(img_as_img)
            img_as_tensor = img_as_tensor.resize_(1,80,80)

        return (img_as_tensor, target)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    foo = CNNData(img_path = "../data_loader/data/")
    bar = foo[1]
    print(bar.shape)
