from MLP import MLP

# Control Variable
filePath = 'Flood_dataset.txt'
layers = [8, 5, 3, 2, 1]
epsilon = 0.001
maxEpoch = 5000
bias = 1


modelA = MLP([8, 4, 1], bias, 0.1, 0.1, maxEpoch, epsilon, filePath, 3)
try:
    modelA.trainModel()
except:
    print(modelA.weights)