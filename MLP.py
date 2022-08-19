from mathOperation import mathOperation
from readFile import readFile
from initialNetwork import initalNetwork
import random

class MLP():

    def __init__(self, layers, bias, learningRate, momentumRate, maxEpoch, epsilon, filePath, startLine) -> None:
        file = readFile(filePath, startLine)
        self.data = file.data
        self.trainingData = file.data
        self.validationData = file.data
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
        self.nodeValue[0] = mathOperation.transpose([self.trainingData[positionOfData]])
        self.inputNode[0] = mathOperation.transpose([self.trainingData[positionOfData]])

        for i in range(len(self.weights)):
            temp = mathOperation.multiplyMatrix(self.weights[i], self.nodeValue[i])
            temp = mathOperation.addBias(temp,self.bias)
            self.inputNode[i + 1] = temp
            self.nodeValue[i + 1] = mathOperation.activationFunc(temp)

        self.nodeValue[0] = mathOperation.transpose([self.trainingData[positionOfData]])
        self.inputNode[0] = mathOperation.transpose([self.trainingData[positionOfData]])

        self.listOfError.append(self.findError(positionOfData))

    def findError(self, positionOfData):
        temp = []
        for i in range(len(self.nodeValue[-1])):
            temp.append(self.desireOutput[positionOfData][i] - self.nodeValue[-1][i][0])
        return temp

    def findsumSquaredErrorAvg(self,isValidation):
        if not isValidation:
            currentData = self.trainingData
        else:
            currentData = self.validationData
        sum = 0
        length = len(currentData) * self.layers[-1]
        for i in range(length):
            for p in self.listOfError[-i-1]:
                sum += p**2
        return sum/length

    def findGrad(self):
        # output
        for outputPosition in range(len(self.grad[-1])):
            self.grad[-1][outputPosition] = [self.listOfError[-1][outputPosition] * mathOperation.diffActivationFunc(self.inputNode[-1][outputPosition])[0]]
        # hidden
        i = len(self.grad) - 2
        while (i >= 0):
            j = 0
            while (j <  len(self.grad[i])):
                self.grad[i][j] = [mathOperation.diffActivationFunc(self.inputNode[i][j])[0] * self.sumOfProductOfWeightAndGrad(i, j)]
                j = j + 1
            i = i - 1

    def sumOfProductOfWeightAndGrad(self, iPosition, jPosition):
        sum = 0
        for k in range(len(self.grad[iPosition + 1])):
            sum += self.weights[iPosition][k][jPosition] * self.grad[iPosition+1][k][0]
        return sum

    def updateWeight(self, epoch):
        if epoch == 0:
            self.weightsChange.append(initalNetwork.initWeight(self))
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weightsChange[-1][i][j][k] = self.learningRate * self.grad[i + 1][j][0] * self.nodeValue[i][k][0]
                        self.weights[i][j][k] += self.weightsChange[-1][i][j][k]
        else:
            self.weightsChange.append(initalNetwork.initWeight(self))
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weightsChange[-1][i][j][k] = self.momentumRate * self.weightsChange[-2][i][j][k] + self.learningRate * self.grad[i + 1][j][0] * self.nodeValue[i][k][0]
                        self.weights[i][j][k] += self.weightsChange[-1][i][j][k]

    def backPropagation(self, epoch):
        self.findGrad()
        self.updateWeight(epoch)

    def trainModel(self):
        epoch = 0
        sumSquaredErrorAvg = 1
        randomListPosition = list(range(0, len(self.trainingData)))
        random.shuffle(randomListPosition)

        while sumSquaredErrorAvg > self.epsilon and epoch < self.maxEpoch:
            for dataPosition in randomListPosition:
                self.feedForward(dataPosition)
                self.backPropagation(epoch)

            sumSquaredErrorAvg = self.findsumSquaredErrorAvg(False)
            epoch += 1
            random.shuffle(randomListPosition)
            print(' SSE AVG:', sumSquaredErrorAvg, ' Epoch:', epoch ,' Error:',self.listOfError[-1])

    def validateFloodDataset(self):
        for p in self.validationData:
            self.feedForward(True)
