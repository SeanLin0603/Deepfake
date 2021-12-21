


DataRoot = '/home/sean/Documents/Forensics/'
ModelDir = './result/'

trainSplitFile = '/home/sean/Documents/Forensics/splits/train.json'
testSplitFile = '/home/sean/Documents/Forensics/splits/test.json'
evalSplitFile = '/home/sean/Documents/Forensics/splits/val.json'


trainBatchSize = 10
testBatchSize = 110

LastEpoch = 0
MaxEpoch = 10

LearningRate = 0.0001
WeightDecay = 0.01

ImageSize = (299, 299)
Norm = [[0.5] * 3, [0.5] * 3]
maxFramePerVideo = 100
