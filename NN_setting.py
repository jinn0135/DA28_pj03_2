import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from torchvision import datasets 
from torchvision import transforms 
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class setting():
    def __init__(self):
        self.trainset, self.testset = None, None
        self.train_set, self.valid_set = None, None

    def getDataset_input(self, trainset, testset):
        self.trainset, self.testset = trainset, testset
        print('trainset length:', len(self.trainset), '/ testset length', len(self.testset))

    def getDataset(self, getdata, dir, dircts, transform):
        if len(dircts)==2: 
            self.trainset = getdata(dir+dircts[0], transform)
            self.testset = getdata(dir+dircts[1], transform)
        else:
            trainset = getdata(dir+dircts[0], transform)
            train_idxs, test_idxs, _, _ = train_test_split(
                range(len(trainset)), trainset.targets, test_size=0.2,
                stratify=trainset.targets, random_state=42)
            self.trainset = Subset(trainset, train_idxs)
            self.testset = Subset(trainset, test_idxs)
        print('trainset length:', len(self.trainset), '/ testset length', len(self.testset))

    def getValid(self, valid_s=0.2):
        train_idxs, valid_idxs, _, _, = train_test_split(
                range(len(self.trainset)), self.trainset.targets, test_size=valid_s,
                stratify=self.trainset.targets, random_state=42)
        print('train length:', len(train_idxs), '/ valid length:', len(valid_idxs))
        self.train_set = Subset(self.trainset, train_idxs)
        self.valid_set = Subset(self.trainset, valid_idxs)
        print(self.train_set[0][0].size(), self.train_set[0][1])
    
    def getDataloader(self, batch_s=16):
        trainloader = DataLoader(self.train_set, batch_size=batch_s, shuffle=True)
        validloader = DataLoader(self.valid_set, batch_size=batch_s, shuffle=True)
        testloader = DataLoader(self.testset, batch_size=batch_s, shuffle=True)
        print('train, valid, test:', len(trainloader), len(validloader), len(testloader))
        return trainloader, validloader, testloader

    def showimg(self, labels_map, data):
        fig, ax = plt.subplots(4,8, figsize=(14,8))
        ax = ax.flatten()
        for i in range(32):
            rand_i = np.random.randint(0, len(data))
            img, label = data[rand_i][0].permute(1,2,0), data[rand_i][1]
            ax[i].axis('off')
            ax[i].imshow(img)
            ax[i].set_title(labels_map[label])

    def showtransimg(self, img):
        fig, ax = plt.subplots(1,5, figsize=(10,3))
        for i, trans in enumerate(['ori','gray','rotate','crop','hori']):
            trans_img = self.transimg(img, trans)
            if trans=='gray': ax[i].imshow(trans_img, cmap='gray')
            else: ax[i].imshow(trans_img)
            ax[i].set_title(trans)
            ax[i].axis('off')

    def transimg(self, img, trans):
        if trans=='ori': return img.permute(1,2,0)
        elif trans=='gray': return transforms.Grayscale()(img).squeeze()
        elif trans=='rotate': return transforms.RandomRotation(degrees=(0,180))(img).permute(1,2,0)
        elif trans=='crop': return transforms.RandomCrop(size=(128,128))(img).permute(1,2,0)
        elif trans=='hori': return transforms.RandomHorizontalFlip(p=0.3)(img).permute(1,2,0)

    
from torch.utils.data import Dataset
import glob
from PIL import Image # Image.open(path)
class createDataset(Dataset):
    # trainset = createDataset(root='cat_and_dog/training_Set/training_set/', transform=transform)
    # trainset.__len__()
    # trainset.__getitem__(5000)[0].size()
    def __init__(self, root, transform):
        self.filepaths = glob.glob(root+'*/*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_filepath = self.filepaths[idx]
        img = Image.open(img_filepath)
        transformed_img = self.transform(img)
        dir_label = img_filepath.split('/')[-2]
        if dir_label == 'cats': label = 0
        else: label = 1
        return transformed_img.label