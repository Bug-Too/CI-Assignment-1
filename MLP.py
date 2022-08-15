from mathOperation import mathOperation
from readFile import readFile
from initialNetwork import initalNetwork
import random


class MLP():

    def __init__(self, layers, bias, learningRate, momentumRate, maxEpoch, epsilon, filePath, startLine) -> None:
        file = readFile(filePath, startLine)
        self.data = file.data
        self.desireOutput = file.desireOutput

        self.layers = layers
        self.bias = bias
        self.maxEpoch = maxEpoch
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.momentumRate = momentumRate

        self.sumSquaredError = 0
        self.listOfError = []

        network = initalNetwork(layers)
        self.weights = network.initWeight()
        self.weightsChange = []
        self.nodeValue = network.initActivation()
        self.inputNode = network.initActivation()
        self.grad = network.initActivation()

    def feedForward(self, positionOfData):
        self.nodeValue[0] = mathOperation.transpose([self.data[positionOfData]])
        self.inputNode[0] = mathOperation.transpose([self.data[positionOfData]])

        for i in range(len(self.weights)):
            temp = mathOperation.multiplyMatrix(self.weights[i], self.nodeValue[i])
            temp = mathOperation.addBias(temp,self.bias)
            self.inputNode[i + 1] = temp
            self.nodeValue[i + 1] = mathOperation.sigmoid(temp)
        self.listOfError.append(self.findError(positionOfData))

    def findError(self, positionOfData):
        temp = []
        for i in range(len(self.nodeValue[-1])):
            temp.append(self.nodeValue[-1][i] - self.desireOutput[positionOfData][i])
        return temp

    def findsumSquaredErrorAvg(self):
        return sum(i * i for i in self.listOfError)

    def findGrad(self, outputPosition):
        # output
        self.grad[-1] = self.listOfError[-1][outputPosition] * mathOperation.diffActivationFunc(self.inputNode[-1])
        # hidden
        i = len(self.grad) - 1
        while (i > 0):
            j = len(self.grad[i])
            while (j > 0):
                self.grad[i][j] = self.sumOfProductOfWeightAndGrad(i, j)
                j = j - 1
            i = i - 1

    def sumOfProductOfWeightAndGrad(self, iPosition, jPosition):
        sum = 0
        for k in range(len(self.grad[iPosition + 1])):
            sum += self.weights[iPosition][jPosition][k] * self.grad[iPosition + 1][k]
        return sum

    def updateWeight(self, epoch):
        if epoch == 1:
            self.weightsChange.append(initalNetwork.initWeight())
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weightsChange[-1][i][j][k] = self.learningRate * self.grad[i + 1][j] * self.nodeValue[i][j]
                        self.weights[i][j][k] += self.weightsChange[-1][i][j][k]
        else:
            self.weightsChange.append(initalNetwork.initWeight())
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weightsChange[-1][i][j][k] = self.momentumRate * self.weightsChange[-2][i][j][
                            k] + self.learningRate * self.grad[i + 1][j] * self.nodeValue[i][j]
                        self.weights[i][j][k] += self.weightsChange[-1][i][j][k]

    def backPropagation(self, epoch):
        for i in range(self.layers[-1]):
            self.findGrad(i)
            self.updateWeight(epoch)

    def trainModel(self):
        floodData = self.data
        epoch = 0
        sumSquaredErrorAvg = 1
        while sumSquaredErrorAvg > self.epsilon and epoch <= self.maxEpoch:
            sumSquaredErrorAvg = self.findsumSquaredErrorAvg()
            for i in range(len(self.data)):
                randomPosition = random.randrange(len(floodData))
                self.feedForward(randomPosition)
                self.backPropagation(epoch)
            sumSquaredErrorAvg = self.findsumSquaredErrorAvg()
            epoch += 1
            print('SSE AVG:', sumSquaredErrorAvg, ' Epoch:', epoch)
