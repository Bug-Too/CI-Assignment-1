from mathOperation import mathOperation
from readFile import readFile
from initialNetwork import initalNetwork

class MLP():



    def __init__(self,layers,bias,learningRate,momentumRate,maxEpoch,epsilon,filePath,startLine) -> None:
        file = readFile(filePath,startLine)
        self.data = file.data
        self.desireOutput = file.desireOutput
        # self.minVal = file.minVal
        # self.maxVal = file.maxVal

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
        self.nodeValue = network.initActivation()
        self.inputNode = network.initActivation()
        self.grad = network.initActivation()




    epsilon = 0.01
    maxEpoch = 5000
    t = 0
    sumSquaredErrorAvg = 0


    def feedForward(self,positionOfData):
        self.nodeValue[0] = self.data[positionOfData]
        self.inputNode[0] = self.data[positionOfData]

        for i in range(len(self.weights-1)):
            temp = mathOperation.multiplyMatrix(self.weights[i],self.nodeValue[i])
            temp = temp + ([self.bias]*len(temp))
            self.inputNode[i+1] = temp
            self.nodeValue[i+1] = mathOperation.sigmoid(temp)
        self.listOfError.append(self.findError(positionOfData))

    def findError(self,positionOfData):
        return self.nodeValue[-1][0] - self.desireOutput[positionOfData]

    def findGrad(self):
        #output
        self.grad[-1] = self.listOfError[-1]*mathOperation.diffActivationFunc(self.inputNode[-1])
        #hidden
        i = len(self.grad) - 1
        while(i>0): 
            j = len(self.grad[i])
            while(j>0):
                self.grad[i][j] = self.sumOfProductOfWeightAndGrad(i,j)
                j = j-1
            i = i-1
        

    def sumOfProductOfWeightAndGrad(self,iPosition,jPosition):
        sum = 0
        for k in range(len(self.grad[iPosition+1])):
            sum += self.weights[iPosition][jPosition][k] * self.grad[iPosition + 1][k]          
        return sum



    def backPropagation(self):
        self.findGrad()

        return