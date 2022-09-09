from MLP import MLP

# Control Variable
filePath = 'cross.pat'
epsilon = 0.03
maxEpoch = 10000
bias = 1
validationRange = 10
listOfSSEAVG = []
output = []
actual = []
for i in range(validationRange):
    modelA = MLP([2, 6, 6, 1], bias, 0.05, 0.05, maxEpoch, epsilon, filePath, i, validationRange)
    modelA.trainModel()
    listOfSSEAVG.append(modelA.sumSquaredErrorAVG)
    output.append(modelA.listOfOutput)
    actual.append(modelA.desireOutput)
print(listOfSSEAVG)
print(output)
print(actual)
