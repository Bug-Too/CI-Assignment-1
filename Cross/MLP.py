from mathOperation import mathOperation
from readFile import readFile
from initialNetwork import initalNetwork
import random


class MLP():

    def __init__(self, layers, bias, learningRate, momentumRate, maxEpoch, epsilon, filePath, validationNumber, validationRange) -> None:
        file = readFile(filePath, validationNumber, validationRange)
        self.data = file.data
        self.trainingData = file.trainingData
        self.validationData = file.validationData
        self.desireOutput = file.desireOutput
        self.validationDesireOutput = file.validationDesireOutput

        self.layers = layers
        self.bias = bias
        self.maxEpoch = maxEpoch
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.momentumRate = momentumRate

        self.sumSquaredErrorAVG = 0
        self.listOfError = []
        self.listOfOutput = []

        network = initalNetwork(layers)
        self.weights = network.initWeight()
        self.weightsChange = []
        self.nodeValue = network.initActivation()
        self.inputNode = network.initActivation()
        self.grad = network.initActivation()

    def feedForward(self, positionOfData, isValidate):
        if isValidate:
            self.nodeValue[0] = mathOperation.transpose([self.validationData[positionOfData]])
            self.inputNode[0] = mathOperation.transpose([self.validationData[positionOfData]])

            for i in range(len(self.weights)):
                temp = mathOperation.multiplyMatrix(self.weights[i], self.nodeValue[i])
                temp = mathOperation.addBias(temp, self.bias)
                self.inputNode[i + 1] = temp
                if i == len(self.weights) - 1:
                    self.nodeValue[i + 1] = mathOperation.outputActivationFunc(temp)
                else:
                    self.nodeValue[i + 1] = mathOperation.hiddenActivationFunc(temp)

            self.nodeValue[0] = mathOperation.transpose([self.validationData[positionOfData]])
            self.inputNode[0] = mathOperation.transpose([self.validationData[positionOfData]])
        else:
            self.nodeValue[0] = mathOperation.transpose([self.trainingData[positionOfData]])
            self.inputNode[0] = mathOperation.transpose([self.trainingData[positionOfData]])

            for i in range(len(self.weights)):
                temp = mathOperation.multiplyMatrix(self.weights[i], self.nodeValue[i])
                temp = mathOperation.addBias(temp, self.bias)
                self.inputNode[i + 1] = temp
                if i == len(self.weights) - 1:
                    self.nodeValue[i + 1] = mathOperation.outputActivationFunc(temp)
                else:
                    self.nodeValue[i + 1] = mathOperation.hiddenActivationFunc(temp)

            self.nodeValue[0] = mathOperation.transpose([self.trainingData[positionOfData]])
            self.inputNode[0] = mathOperation.transpose([self.trainingData[positionOfData]])

        self.listOfError.append(self.findError(positionOfData, isValidate))

    def findError(self, positionOfData, isValidate):
        if isValidate:
            temp = []
            for i in range(len(self.nodeValue[-1])):
                temp.append(self.validationDesireOutput[positionOfData][i] - round(self.nodeValue[-1][i][0]))
            return temp
        else:
            temp = []
            for i in range(len(self.nodeValue[-1])):
                temp.append(self.desireOutput[positionOfData][i] - round(self.nodeValue[-1][i][0]))
            return temp

    def findsumSquaredErrorAvg(self, isValidation):
        if not isValidation:
            currentData = self.trainingData
        else:
            currentData = self.validationData
        sum = 0
        for i in range(len(currentData)):
            for p in self.listOfError[-i - 1]:
                sum += p ** 2
        return sum / len(currentData)

    def findGrad(self):
        # output
        for outputPosition in range(len(self.grad[-1])):
            self.grad[-1][outputPosition] = [self.listOfError[-1][outputPosition] * mathOperation.outputDiffActivationFunc(self.inputNode[-1][outputPosition])[0]]
        # hidden
        i = len(self.grad) - 2
        while (i >= 0):
            j = 0
            while (j < len(self.grad[i])):
                self.grad[i][j] = [
                    mathOperation.hiddenDiffActivationFunc(self.inputNode[i][j])[0] * self.sumOfProductOfWeightAndGrad(i, j)]
                j = j + 1
            i = i - 1

    def sumOfProductOfWeightAndGrad(self, iPosition, jPosition):
        sum = 0
        for k in range(len(self.grad[iPosition + 1])):
            sum += self.weights[iPosition][k][jPosition] * self.grad[iPosition + 1][k][0]
        return sum

    def updateWeight(self, epoch):
        if epoch == 0:
            self.weightsChange.append(initalNetwork.initWeight(self))
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weightsChange[-1][i][j][k] = self.learningRate * self.grad[i + 1][j][0] * \
                                                          self.nodeValue[i][k][0]
                        self.weights[i][j][k] += self.weightsChange[-1][i][j][k]
        else:
            self.weightsChange.append(initalNetwork.initWeight(self))
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weightsChange[-1][i][j][k] = self.momentumRate * self.weightsChange[-2][i][j][
                            k] + self.learningRate * self.grad[i + 1][j][0] * self.nodeValue[i][k][0]
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
                self.feedForward(dataPosition, False)
                self.backPropagation(epoch)

            sumSquaredErrorAvg = self.findsumSquaredErrorAvg(False)
            epoch += 1
            random.shuffle(randomListPosition)
            print(' SSE AVG:', sumSquaredErrorAvg, ' Epoch:', epoch, ' Error:', self.listOfError[-1])
        self.validate()

    def validate(self):
        for i in range(len(self.validationData)):
            self.feedForward(i, True)
            self.listOfOutput.append(round(self.desireOutput[i][0]))
        self.sumSquaredErrorAVG = self.findsumSquaredErrorAvg(True)
        print('Validate SSE AVG: ', self.sumSquaredErrorAVG)
