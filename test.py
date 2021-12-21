import numpy as np
import os
import random
from scipy.io import savemat
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# from dataset import Dataset
from dataset_ff import Dataset
from xception import Model
import config

def process_batch(batch):
    img = batch['img']
    lab = batch['lab']

    x = net.model(img)
    loss = lossFunction(x, lab)
    pred = torch.max(x, dim=1)[1]
    acc = (pred == lab).float().mean()
    res = { 'lab': lab, 'score': x, 'pred': pred }
    results = {}
    for r in res:
        results[r] = res[r].squeeze().cpu().numpy()
    return { 'loss': loss, 'acc': acc }, results

def run_epoch(index):
    print('Epoch: {0}'.format(index))

    realTestLoader = testData.datasets[0].loader
    fakeTestLoader = testData.datasets[1].loader

    step = 0
    # Real
    for batch in realTestLoader:
        img = batch['img'].cuda()
        lab = batch['lab'].cuda()
        input = { 'img':img, 'lab':lab}

        net.model.eval()
        with torch.no_grad():
            losses, results = process_batch(input)

            savemat('{0}{1}_{2}.mat'.format(resultDir, 0, step), results)

            if step % 10 == 0:
                print('{0} - '.format(step) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]))

        step = step + 1

    step = 0
    # Fake
    for batch in fakeTestLoader:
        img = batch['img'].cuda()
        lab = batch['lab'].cuda()
        input = { 'img':img, 'lab':lab}

        net.model.eval()
        with torch.no_grad():
            losses, results = process_batch(input)

            savemat('{0}{1}_{2}.mat'.format(resultDir, 1, step), results)

            if step % 10 == 0:
                print('{0} - '.format(step) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]))

        step = step + 1

if __name__ == "__main__":
    torch.backends.deterministic = True
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("[Info] Loading dataset")
    testData = Dataset('test', config.testBatchSize, config.ImageSize, config.Norm)
    print("[Info] Built dataset\n\n")

    net = Model(load_pretrain=False)
    net.model.cuda()
    lossFunction = nn.CrossEntropyLoss().cuda()

    for epoch in range(config.MaxEpoch):
        resultDir = '{0}{1}/'.format(config.ModelDir, epoch)

        if not os.path.exists(resultDir):
            print('[Info] Create foler: {}'.format(resultDir))
            os.makedirs(resultDir, exist_ok=True)

        net.load(epoch, config.ModelDir)
        run_epoch(epoch)

    print('Test completed.')
