import os
import pandas as pd
from imutils import paths
from imageio import imread
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from utils import get_files_from_split

RealSets = ['RealFF']
FakeSets = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

class DatasetInstance:
    def __init__(self, labelName, label, dataType, fileList, imageSize, norm, batchSize):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ])
        
        self.labelName = labelName
        self.label = label
        self.dataType = dataType
        self.imageSize = imageSize

        # self.imagePaths = []
        totalImages = []

        if labelName == 'real':
            # Real
            for subset in RealSets:
                subsetFolder = os.path.join(config.DataRoot, subset, 'c23', 'images')
                # print(subsetFolder)

                for videos in fileList:
                    frameFolder = os.path.join(subsetFolder, videos)
                    totalImages.append(frameFolder)
                    # print('{}, {}, {}'.format(labelName, frameFolder, os.path.exists(frameFolder)))
        else:
            # Fake
            for subset in FakeSets:
                subsetFolder = os.path.join(config.DataRoot, subset, 'c23', 'images')
                # print(subsetFolder)

                for videos in fileList:
                    frameFolder = os.path.join(subsetFolder, videos)
                    totalImages.append(frameFolder)
                    # print('{}, {}, {}'.format(labelName, frameFolder, os.path.exists(frameFolder)))

        self.images = []
        self.dataDir = "{0}{1}/{2}".format(config.DataRoot, self.dataType, self.labelName)
        
        for folder in totalImages:
            frames = list(paths.list_images(folder))
            frameNum = len(frames)
            
            # print('[Info] size of {} : {}'.format(folder, frameNum))

            if frameNum <= config.maxFramePerVideo:
                interval = 1
            else:
                interval = int(frameNum / config.maxFramePerVideo)

            # print('frameNum: {}'.format(frameNum))
            # print('interval: {}'.format(interval))

            for i in range(0, frameNum, interval):
                self.images.append(frames[i])

        # read entire folder
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
    def __init__(self, dataType, batchSize, imageSize, norm):
        
        if dataType == 'train':
            splitData = pd.read_json(config.trainSplitFile, dtype=False)
        elif dataType == 'test':
            splitData = pd.read_json(config.testSplitFile, dtype=False)
        elif dataType == 'eval':
            splitData = pd.read_json(config.evalSplitFile, dtype=False)

        realVideos, fakeVideos = get_files_from_split(splitData)
        realVideos = sorted(realVideos)
        fakeVideos = sorted(fakeVideos)

        # print('[Info] realVideos: {}'.format(len(realVideos)))
        # print('[Info] fakeVideos: {}'.format(len(fakeVideos)))

        realData = DatasetInstance('real', 0, dataType, realVideos, imageSize, norm, batchSize)
        fakeData = DatasetInstance('fake', 1, dataType, fakeVideos, imageSize, norm, batchSize)
        
        self.datasets = [realData, fakeData]