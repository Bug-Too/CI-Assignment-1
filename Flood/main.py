from MLP import MLP
from mathOperation import mathOperation

# Control Variable
filePath = 'Flood_dataset.txt'
epsilon = 0.0006
maxEpoch = 5000
bias = 1
validationRange = 10
listOfSSEAVG = []

for i in range(validationRange):
    modelA = MLP([8, 4, 2, 1], bias, 0.05, 0.05, maxEpoch, epsilon, filePath, 3, i, validationRange)
    modelA.trainModel()
    listOfSSEAVG.append(modelA.sumSquaredErrorAVG)
print(mathOperation.deNormData(listOfSSEAVG, modelA.minVal, modelA.maxVal))
print(listOfSSEAVG)



