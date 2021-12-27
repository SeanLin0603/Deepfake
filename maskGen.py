import os
from pathlib import Path
from imageio import imread
from torchvision import transforms
from torchvision.utils import save_image

transform_norm3 = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((299, 299)), 
    transforms.ToTensor()])


if __name__ == '__main__':

    originalDir = '/home/sean/Documents/faceforensic/frame/val/original'
    alteredDir = '/home/sean/Documents/faceforensic/frame/val/altered'
    maskDir = '/home/sean/Documents/faceforensic/frame/val/mask'

    realFolders = os.listdir(originalDir)
    realFolders.sort()
    fakeFolders = os.listdir(alteredDir)
    fakeFolders.sort()
    realFolderNum = len(realFolders)
    fakeFolderNum = len(fakeFolders)

    print('[Info] realFolderNum: {}'.format(realFolderNum))
    print('[Info] fakeFolderNum: {}\n'.format(fakeFolderNum))

    for i in range(100, realFolderNum):
        realFolder = os.path.join(originalDir, realFolders[i])
        fakeFolder = os.path.join(alteredDir, realFolders[i])
        folderExist = os.path.isdir(realFolder) and os.path.isdir(fakeFolder)

        # print(realFolder)
        # print(fakeFolder)

        if folderExist:
            realImages = list(Path(realFolder).rglob('*.*'))
            realImages.sort()
            fakeImages = list(Path(fakeFolder).rglob('*.*'))
            fakeImages.sort()
            realImageNum = len(realImages)
            fakeImageNum = len(fakeImages)

            print('[Info] realImageNum: {}'.format(realImageNum))
            print('[Info] fakeImageNum: {}\n'.format(fakeImageNum))

            for j in range(realImageNum):
                realImagePath = str(realImages[j])
                fakeImagePath = str(fakeImages[j])
                
                realImage = transform_norm3(imread(realImagePath))
                fakeImage = transform_norm3(imread(fakeImagePath))
                mask = abs(fakeImage - realImage)

                tokens = realImagePath.split('/')
                tokens[7] = 'mask'
                maskImagePath = '/'.join(tokens)
                
                saveDir = os.path.dirname(maskImagePath)
                
                if not os.path.isdir(saveDir):
                    print('[Info] Create: {}\n'.format(saveDir))
                    os.makedirs(saveDir, mode=755)

                save_image(mask, maskImagePath)
                os.chmod(maskImagePath, 0o755)
                # print('[Info] realImagePath: {}'.format(realImagePath))
                # print('[Info] fakeImagePath: {}'.format(fakeImagePath))
                print('[Info] {}/{}, maskImagePath: {}\n'.format(i, realFolderNum, maskImagePath))

                # print(realImage)
                # print(fakeImage)


        else:
            print("[Info] Folder does not match: #{}, {}, {}".format(i, realFolder, fakeFolder, folderExist))
        