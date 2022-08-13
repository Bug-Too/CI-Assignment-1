from mathOperation import mathOperation
from readFile import readFile
from initialNetwork import initalNetwork

class MLP():



    def __init__(self,layers,weights,bias,maxEpoch,epsilon) -> None:
        self.layers = layers
        self.weights = weights
        self.bias = bias
        self.maxEpoch = maxEpoch
        self.epsilon = epsilon



    epsilon = 0.01
    maxEpoch = 5000
    t = 0
    sumSquaredErrorAvg = 0


    def feedForward(data,bias):
        epouch = 0
        sumSquaredError = 0
        for j in range(len(data)):
            epouch += 1
            for i in range(len(weights)):
                activationVal[0] = data[j]
                VVal[0] = data[j]
                i += 1
                v = multiplyMatrix(weights[i],activationVal[i])
                v = v+([bias]*len(v))
                VVal[i+1] = v
                activationVal[i+1] = sigmoid(v)
            sumSquaredError += (desireOutput[j] - activationVal[-1][0])**2
        return sumSquaredError
    
    def backPropagation(self):
        return