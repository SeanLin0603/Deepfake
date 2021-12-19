import glob, os
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.arrayprint import printoptions
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import auc

import config

MASK_THRESHOLD = 0.5
saveDir = 'roc'

# Compile the results into a single variable for processing
def compute_result_file(rfn):
  rf = loadmat(rfn)
  res = {}
  for r in ['lab', 'score', 'pred']:
    res[r] = rf[r].squeeze()
  return res

def eval(epoch):

  if not os.path.isdir(saveDir):
    os.mkdir(saveDir)

  text = open(saveDir +'/'+ str(epoch)+'.txt', 'w')

#   RESDIR = './models/xcp_tmp/results/' + str(epoch) + '/'
  RESDIR = './result/' + str(epoch) + '/'
  print('[Info] Result dir: {}'.format(RESDIR))
  RESFILENAMES = glob.glob(RESDIR + '*.mat')
  fileNums = len(RESFILENAMES)

  print('{0} result files'.format(fileNums), file=text)


  rf = compute_result_file(RESFILENAMES[0])
  TOTAL_RESULTS = {'lab': rf['lab'], 'score': rf['score'], 'pred': rf['pred']}

  for i in range(fileNums):
    print("\rLoading... Epoch:{}, #{}/{}".format(epoch, i, fileNums), end=' ')
    rfn = RESFILENAMES[i] 
    rf = compute_result_file(rfn)

    rf['lab'] = rf['lab'].reshape((rf['lab'].size,))
    # print("[Info] rf['lab']:{}".format(rf['lab'].shape))
    # print("[Info] TOTAL_RESULTS['lab']:{}".format(TOTAL_RESULTS['lab'].shape))
    TOTAL_RESULTS['lab'] = np.concatenate((TOTAL_RESULTS['lab'], rf['lab']), axis=0)
    
    rf['score'] = rf['score'].reshape((int(rf['score'].size/2), 2))
    # print("[Info] rf['score']:{}".format(rf['score'].shape))
    # print("[Info] TOTAL_RESULTS['score']:{}".format(TOTAL_RESULTS['score'].shape))
    TOTAL_RESULTS['score'] = np.concatenate((TOTAL_RESULTS['score'], rf['score']), axis=0)
    
    rf['pred'] = rf['pred'].reshape((rf['pred'].size,))
    # print("[Info] rf['pred']:{}".format(rf['pred'].shape))
    # print("[Info] TOTAL_RESULTS['pred']:{}".format(TOTAL_RESULTS['pred'].shape))
    TOTAL_RESULTS['pred'] = np.concatenate((TOTAL_RESULTS['pred'], rf['pred']), axis=0)

  print('')
  print(' ', file=text)
  print('Found {0} total images with scores.'.format(TOTAL_RESULTS['lab'].shape[0]), file=text)
  print('  {0} results are real images'.format((TOTAL_RESULTS['lab'] == 0).sum()), file=text)
  print('  {0} results are fake images'.format((TOTAL_RESULTS['lab'] == 1).sum()), file=text)
  #for r in TOTAL_RESULTS:
  #  print('{0} has shape {1}'.format(r, TOTAL_RESULTS[r].shape))

  # Compute the performance numbers
  PRED_ACC = (TOTAL_RESULTS['lab'] == TOTAL_RESULTS['pred']).astype(np.float32).mean()

  FPR, TPR, THRESH = metrics.roc_curve(TOTAL_RESULTS['lab'], TOTAL_RESULTS['score'][:,1], drop_intermediate=False)
  AUC = auc(FPR, TPR)
  FNR = 1 - TPR
  EER = FNR[np.argmin(np.absolute(FNR - FPR))]
  TPR_AT_FPR_NOT_0 = TPR[FPR != 0].min()
  TPR_AT_FPR_THRESHOLDS = {}
  for t in range(-1, -7, -1):
    thresh = 10**t
    TPR_AT_FPR_THRESHOLDS[thresh] = TPR[FPR <= thresh].max()

  # Print out the performance numbers
  print('Prediction Accuracy: {0:.4f}'.format(PRED_ACC), file=text)
  print('AUC: {0:.4f}'.format(AUC), file=text)
  print('EER: {0:.4f}'.format(EER), file=text)
  print('Minimum TPR at FPR != 0: {0:.4f}'.format(TPR_AT_FPR_NOT_0), file=text)

  print('TPR at FPR Thresholds:', file=text)
  for t in TPR_AT_FPR_THRESHOLDS:
    print('  {0:.10f} TPR at {1:.10f} FPR'.format(TPR_AT_FPR_THRESHOLDS[t], t), file=text)

  # draw the ROC curve
  fig = plt.figure()
  plt.plot(FPR, TPR)
  plt.xlabel('FPR (%)')
  plt.ylabel('TPR (%)')
  plt.xscale('log')
  plt.xlim([10e-8,1])
  plt.ylim([0, 1])
  plt.grid()
  # plt.show()
  plt.savefig(saveDir +'/'+ str(epoch) + '.png', dpi=300, bbox_inches='tight')

  text.close()

if __name__ == '__main__':

  for i in range(config.MaxEpoch):
    eval(i)

