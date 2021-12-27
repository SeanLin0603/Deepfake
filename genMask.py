from genericpath import exists
import os
from pathlib import Path
from imageio import imread
from torchvision import transforms
from torchvision.utils import save_image

from dataset_ff import FakeSets, RealSets
from eval import saveROCImage

transform_norm3 = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((299, 299)), 
    transforms.ToTensor()])

RealSets = ['RealFF']
FakeSets = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
Root = '/home/sean/Documents/Forensics/'
postfix = 'c23/images/'
# postfix = 'c23/original_images/'

if __name__ == '__main__':

    realDir = os.path.join(Root, RealSets[0], postfix)
    dfDir = os.path.join(Root, FakeSets[0], postfix)
    f2fDir = os.path.join(Root, FakeSets[1], postfix)
    fsDir = os.path.join(Root, FakeSets[2], postfix)
    ntDir = os.path.join(Root, FakeSets[3], postfix)
    
    # maskDir = '/home/sean/Documents/faceforensic/frame/val/mask'

    # videos
    realFolders = os.listdir(realDir)
    dfFolders = os.listdir(dfDir)
    f2fFolders = os.listdir(f2fDir)
    fsFolders = os.listdir(fsDir)
    ntFolders = os.listdir(ntDir)

    realFolders.sort()
    dfFolders.sort()
    f2fFolders.sort()
    fsFolders.sort()
    ntFolders.sort()

    realFolderNum = len(realFolders)
    dfFolderNum = len(dfFolders)
    f2fFolderNum = len(f2fFolders)
    fsFolderNum = len(fsFolders)
    ntFolderNum = len(ntFolders)

    print('[Info] realFolderNum: {}'.format(realFolderNum))
    print('[Info] dfFolderNum: {}'.format(dfFolderNum))
    print('[Info] f2fFolderNum: {}'.format(f2fFolderNum))
    print('[Info] fsFolderNum: {}'.format(fsFolderNum))
    print('[Info] ntFolderNum: {}'.format(ntFolderNum))

    same = 0
    diff = 0
    
    # frames
    for i in range(745, 1000):
        realFramePath = os.path.join(realDir, realFolders[i])
        dfFramePath = os.path.join(dfDir, dfFolders[i])
        f2fFramePath = os.path.join(f2fDir, f2fFolders[i])
        fsFramePath = os.path.join(fsDir, fsFolders[i])
        ntFramePath = os.path.join(ntDir, ntFolders[i])
        
        realFrames = os.listdir(realFramePath)
        dfFrames = os.listdir(dfFramePath)
        f2fFrames = os.listdir(f2fFramePath)
        fsFrames = os.listdir(fsFramePath)
        ntFrames = os.listdir(ntFramePath)

        realFrames.sort()
        dfFrames.sort()
        f2fFrames.sort()
        fsFrames.sort()
        ntFrames.sort()

        realFrameNum = len(realFrames)
        dfFrameNum = len(dfFrames)
        f2fFrameNum = len(f2fFrames)
        fsFrameNum = len(fsFrames)
        ntFrameNum = len(ntFrames)

        minFrameNum = min(realFrameNum, dfFrameNum, f2fFrameNum, fsFrameNum, ntFrameNum)
        # print('minFrameNum: {}'.format(minFrameNum))

        ################################ 
        # processPath = realFramePath
        # processPath = dfFramePath
        # processPath = f2fFramePath
        # processPath = fsFramePath
        processPath = ntFramePath
        ################################ 
        savePath = processPath.replace('Forensics', 'FF_mask')
        # print(processPath)
        # print(dstPath)

        if not os.path.exists(savePath):
            print('[Info] Create directory: {}'.format(savePath))
            os.makedirs(savePath, exist_ok=True)


        for j in range(1, minFrameNum):
            realImgPath = '%s/%05d.jpg' % (realFramePath, j)
            fakeImgPath = '%s/%05d.jpg' % (processPath, j)
            cpPath = '%s/%05d.jpg' % (savePath, j)
            command = 'cp {} {}'.format(fakeImgPath, cpPath)

            maskPath = cpPath.replace('images', 'mask')
            maskFolder = os.path.dirname(maskPath)
            if not os.path.exists(maskFolder):
                print('[Info] Create directory: {}'.format(maskFolder))
                os.makedirs(maskFolder)

            print(maskPath)

            if not (os.path.exists(realImgPath) and os.path.exists(fakeImgPath)):
                continue

            os.system(command)
            # print(command)

            realImage = transform_norm3(imread(realImgPath))
            fakeImage = transform_norm3(imread(fakeImgPath))
            mask = abs(fakeImage - realImage)
            save_image(mask, maskPath)





        # if dfSubFolderNum == realSubFolderNum and f2fSubFolderNum == realSubFolderNum and fsSubFolderNum == realSubFolderNum and ntSubFolderNum == realSubFolderNum:
        #     same = same + 1
        # else:
        #     diff = diff + 1

        # print("{}, {}".format(realFolder, realSubFolderNum))
        # print("{}, {}".format(dfFolder, dfSubFolderNum))
        # print("{}, {}".format(f2fFolder, f2fSubFolderNum))
        # print("{}, {}".format(fsFolder, fsSubFolderNum))
        # print("{}, {}".format(ntFolder, ntSubFolderNum))
        print()

    print('Same: {}, Diff: {}'.format(same, diff))
        
