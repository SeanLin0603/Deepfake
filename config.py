
PretrainWeight = './weights/xception-b5690688.pth'

DataRoot = '/home/sean/Documents/Forensics/'
ModelDir = './result/'

TrainSplitFile = '/home/sean/Documents/Forensics/splits/train.json'
TestSplitFile = '/home/sean/Documents/Forensics/splits/test.json'
EvalSplitFile = '/home/sean/Documents/Forensics/splits/val.json'

TrainBatchSize = 10
TestBatchSize = 110

LastEpoch = 0
MaxEpoch = 10

LearningRate = 0.0001
WeightDecay = 0.01

ImageSize = (299, 299)
Norm = [[0.5] * 3, [0.5] * 3]
MaxFramePerVideo = 100
