import glob, os
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.arrayprint import printoptions
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import auc

import config

MASK_THRESHOLD = 0.5

def saveROCImage(fpr, tpr, imagePath):
  # draw the ROC curve
  fig = plt.figure()
  plt.plot(fpr, tpr)
  plt.xlabel('FPR (%)')
  plt.ylabel('TPR (%)')
  plt.xscale('log')
  plt.xlim([10e-8,1])
  plt.ylim([0, 1])
  plt.grid()
  # plt.show()
  plt.savefig(imagePath, dpi=300, bbox_inches='tight')

# Compile the results into a single variable for processing
def compute_result_file(rfn):
  rf = loadmat(rfn)
  res = {}
  for r in ['lab', 'score', 'pred']:
    res[r] = rf[r].squeeze()
  return res

def eval(epoch):

  # Open a text file for logging
  txtPath = '{}/{}.txt'.format(saveDir, epoch)
  logFile = open(txtPath, 'w')

  testPath = os.path.join(config.ModelDir, str(epoch))
  print('[Info] Result dir: {}'.format(testPath))
  if not os.path.exists(testPath):
    print('[Info] Directory do not exist: {}'.format(testPath))
    return
  
  testFiles = glob.glob(testPath + '/*.mat')
  testNum = len(testFiles)

  rf = compute_result_file(testFiles[0])
  result = {'lab': rf['lab'], 'score': rf['score'], 'pred': rf['pred']}

  for i in range(testNum):
    print("\rLoading... Epoch:{}, #{}/{}".format(epoch, i, testNum), end=' ')
    
    rfn = testFiles[i] 
    rf = compute_result_file(rfn)
    rf['lab'] = rf['lab'].reshape((rf['lab'].size,))
    rf['score'] = rf['score'].reshape((int(rf['score'].size/2), 2))
    rf['pred'] = rf['pred'].reshape((rf['pred'].size,))
    
    result['lab'] = np.concatenate((result['lab'], rf['lab']), axis=0)
    result['score'] = np.concatenate((result['score'], rf['score']), axis=0)
    result['pred'] = np.concatenate((result['pred'], rf['pred']), axis=0)
    
    # print("[Info] rf['lab']:{}".format(rf['lab'].shape))
    # print("[Info] result['lab']:{}".format(result['lab'].shape))
    # print("[Info] rf['score']:{}".format(rf['score'].shape))
    # print("[Info] result['score']:{}".format(result['score'].shape))
    # print("[Info] rf['pred']:{}".format(rf['pred'].shape))
    # print("[Info] result['pred']:{}".format(result['pred'].shape))
  print('')

  # Compute the performance numbers
  acc = (result['lab'] == result['pred']).astype(np.float32).mean()

  fpr, tpr, threshold = metrics.roc_curve(result['lab'], result['score'][:,1], drop_intermediate=False)
  areaUnderCurve = auc(fpr, tpr)
  fnr = 1 - tpr
  eer = fnr[np.argmin(np.absolute(fnr - fpr))]
  TPR_AT_FPR_NOT_0 = tpr[fpr != 0].min()
  TPR_AT_FPR_THRESHOLDS = {}
  for t in range(-1, -7, -1):
    thresh = 10**t
    TPR_AT_FPR_THRESHOLDS[thresh] = tpr[fpr <= thresh].max()

  # Export
  print('{0} result files\n\n'.format(testNum), file=logFile)
  print('Found {0} total images with scores.'.format(result['lab'].shape[0]), file=logFile)
  print('  {0} results are real images'.format((result['lab'] == 0).sum()), file=logFile)
  print('  {0} results are fake images'.format((result['lab'] == 1).sum()), file=logFile)

  print('Prediction Accuracy: {0:.4f}'.format(acc), file=logFile)
  print('AUC: {0:.4f}'.format(areaUnderCurve), file=logFile)
  print('EER: {0:.4f}'.format(eer), file=logFile)
  print('Minimum TPR at FPR != 0: {0:.4f}'.format(TPR_AT_FPR_NOT_0), file=logFile)

  print('TPR at FPR Thresholds:', file=logFile)
  for t in TPR_AT_FPR_THRESHOLDS:
    print('  {0:.10f} TPR at {1:.10f} FPR'.format(TPR_AT_FPR_THRESHOLDS[t], t), file=logFile)

  imgPath = "{}/{}.png".format(saveDir, epoch)
  saveROCImage(fpr, tpr, imgPath)

  logFile.close()

if __name__ == '__main__':

  for epoch in range(config.MaxEpoch):
    
    # Create evaluation result directory
    saveDir = os.path.join(config.ModelDir, 'roc')
    # print('[Info] saveDir: {}'.format(saveDir))
    if not os.path.isdir(saveDir):
      os.mkdir(saveDir)

    eval(epoch)
  
  print('Eval completed.')


