import os
import numpy as np
import random
import torch
from torch._C import iinfo
import torch.optim as optim
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# from dataset import Dataset
from dataset_ff import Dataset
from xception import Model
import config


def process_batch(batch, mode):
    if mode == 'train':
        net.model.train()
        
        image = batch['img']
        label = batch['lab']
        pred = net.model(image)

        # calc loss
        # print('[Info] pred: {}'.format(pred))
        # print('[Info] label: {}'.format(label))
        loss = lossFunction(pred, label)

        pred = torch.max(pred, dim=1)[1]
        acc = (pred == label).float().mean()
        losses = { 'loss': loss, 'acc': acc }
        
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()

    return losses


def run_epoch(index):
    print('Epoch: {0}'.format(index))
    step = 0

    realTrainLoader = trainData.datasets[0].loader
    fakeTrainLoader = trainData.datasets[1].loader

    for batch in zip(realTrainLoader, fakeTrainLoader):
        batch = list(batch)

        img = torch.cat([_['img'] for _ in batch], dim=0).cuda()
        lab = torch.cat([_['lab'] for _ in batch], dim=0).cuda()
        #im_name = torch.cat([_['im_name'] for _ in batch], dim=0)
        input = { 'img': img, 'lab': lab }
        # print(input)

        losses = process_batch(input, 'train')

        if step % 10 == 0:
            print('\r{0} - '.format(step) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
        if step % 100 == 0:
            print('\n', end='')

        step = step + 1
    net.save(epoch, optimizer, config.ModelDir)

if __name__ == "__main__":
    
    torch.backends.deterministic = True
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("[Info] Loading dataset")
    trainData = Dataset('train', config.trainBatchSize, config.ImageSize, config.Norm)
    print("[Info] Built dataset\n\n")

    net = Model(load_pretrain=True)
    optimizer = optim.Adam(net.model.parameters(), lr=config.LearningRate, weight_decay=config.WeightDecay)
    lossFunction = nn.CrossEntropyLoss().cuda()
    # lossFunction = nn.BCELoss().cuda()
    net.model.cuda()
    # print(net.model)
    
    for epoch in range(config.LastEpoch, config.MaxEpoch):
        run_epoch(epoch)
    print('Train completed.')
    