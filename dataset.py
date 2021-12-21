import os
import random
from imutils import paths
from imageio import imread
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config

class DatasetInstance:
    def __init__(self, labelName, label, dataType, imageSize, norm, seed, batchSize):
        self.imageSize = imageSize
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ])
        
        self.labelName = labelName
        self.label = label
        self.dataType = dataType
        self.images = []
        self.dataDir = "{0}{1}/{2}".format(config.DataRoot, self.dataType, self.labelName)
        
        videoFolders = os.listdir(self.dataDir)
        for folder in videoFolders:
            path = os.path.join(self.dataDir, str(folder))
            frames = list(paths.list_images(path))
            length = len(frames)
            
            maxFrame = 100
            if length <= maxFrame:
                interval = 1
            else:
                interval = int(length / maxFrame)

            # print('Length: {}'.format(length))
            # print('interval: {}'.format(interval))

            for i in range(0, length, interval):
                self.images.append(frames[i])

            # print(len(self.images))

        # self.images = list(paths.list_images(self.dataDir))
        
        self.loader = DataLoader(self, num_workers=8, batch_size=batchSize, shuffle=(self.dataType != 'test'), pin_memory=False)
        print('[Info] Constructed dataset: {0} of size {1}'.format(self.dataDir, self.__len__()))

    def __getitem__(self, index):
        imageName = self.images[index]
        image = self.transform(imread(imageName))
        return {'img': image, 'lab': self.label, 'imageName':imageName}

    def __len__(self):
        return len(self.images)

class Dataset:
    def __init__(self, dataType, batchSize, imageSize, norm, seed):
        realData = DatasetInstance('real', 0, dataType, imageSize, norm, seed, batchSize)
        fakeData = DatasetInstance('fake', 1, dataType, imageSize, norm, seed, batchSize)
        
        self.datasets = [realData, fakeData]