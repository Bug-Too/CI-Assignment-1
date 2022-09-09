import math

class mathOperation():
    @staticmethod
    def exp(x):
        e = 2.718281828459045
        if(x<=700):
            return e**x
        else:
            return math.inf
        

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + mathOperation.exp(0 - x))

    @staticmethod
    def sigmoidPrime(x):
        return (1.0 / (1.0 + mathOperation.exp(-x)))*(1.0 - (1.0 / (1.0 + mathOperation.exp(-x))))

    @staticmethod
    def leakyRelu(x):
        if x < 0:
            return 0.01 * x
        else:
            return x

    @staticmethod
    def leakyReluPrime(x):
        if x <= 0:
            return 0.01
        else:
            return 1

    @staticmethod
    def activationFunc(list):
        temp = []
        for v in list:
            temp.append([mathOperation.leakyRelu(v[0])])
        return temp

    @staticmethod
    def diffActivationFunc(list):
        temp = []
        for v in list:
            temp.append(mathOperation.leakyReluPrime(v))
        return temp

    @staticmethod
    def multiplyMatrix(X,Y):
        return [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]

    @staticmethod
    def normData(list,minVal,maxVal):
        temp =[]
        for n in list:
            temp.append((n-minVal)/(maxVal-minVal))
        return temp

    @staticmethod
    def deNormData(list, minVal, maxVal):
        temp = []
        for n in list:
            temp.append(n*(maxVal-minVal)+minVal)
        return temp

    @staticmethod
    def transpose(list):
        return [[list[j][i] for j in range(len(list))] for i in range(len(list[0]))]

    @staticmethod
    def addBias(list,bias):
        temp = []
        for n in list:
            temp.append([n[0]+bias])
        return temp
