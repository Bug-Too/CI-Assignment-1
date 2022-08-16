from MLP import MLP

# Control Variable
filePath = 'Flood_dataset.txt'
layers = [8, 5, 3, 2, 1]
epsilon = 0.01
maxEpoch = 5000
bias = 1


modelA = MLP([8, 5, 1], bias, 0.1, 0.05, maxEpoch, epsilon, filePath, 3)
modelA.trainModel()
