from MLP import MLP

# Control Variable
filePath = 'Flood_dataset.txt'
layers = [8, 5, 3, 2, 1]
epsilon = 0.0005
maxEpoch = 50000
bias = 1


modelA = MLP([8, 4, 2, 1], bias, 0.05, 0.05, maxEpoch, epsilon, filePath, 3)
modelA.trainModel()
print(modelA.weights)
