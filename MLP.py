from mathOperation import mathOperation
from readFile import readFile
from initialNetwork import initalNetwork

class MLP():



    def __init__(self,layers,bias,maxEpoch,epsilon,filePath,startLine) -> None:
        file = readFile(filePath,startLine)
        self.data = file.data
        self.desireOutput = file.desireOutput
        # self.minVal = file.minVal
        # self.maxVal = file.maxVal

        self.layers = layers
        self.bias = bias
        self.maxEpoch = maxEpoch
        self.epsilon = epsilon

        network = initalNetwork(layers)
        self.weights = network.initWeight()
        self.nodeValue = network.initActivation()
        self.inputNode = network.initActivation()




    epsilon = 0.01
    maxEpoch = 5000
    t = 0
    sumSquaredErrorAvg = 0


    def feedForward(self):
        epouch = 0
        sumSquaredError = 0
        for j in range(len(self.data)):
            epouch += 1
            for i in range(len(self.weights)):
                self.nodeValue[0] = self.data[j]
                self.inputNode[0] = self.data[j]
                i += 1
                v = mathOperation.multiplyMatrix(self.weights[i],self.nodeValue[i])
                v = v+([self.bias]*len(v))
                self.inputNode[i+1] = v
                self.nodeValue[i+1] = mathOperation.sigmoid(v)
            sumSquaredError += ((self.desireOutput[j] - self.nodeValue[-1][0])**2)
        return sumSquaredError
    
    def backPropagation(self):
        return