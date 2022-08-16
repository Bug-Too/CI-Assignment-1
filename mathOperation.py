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
        temp = []
        for v in x:
            temp.append([1.0 / (1.0 + mathOperation.exp(-v[0]))])
        return temp

    @staticmethod
    def sigmoidPrime(x):
        temp = []
        for v in x:
            temp.append((1.0 / (1.0 + mathOperation.exp(-v)))*(1.0 - (1.0 / (1.0 + mathOperation.exp(-v)))))
        return temp

    @staticmethod
    def activationFunc(list):
        return mathOperation.sigmoid(list)

    @staticmethod
    def diffActivationFunc(list):
        return mathOperation.sigmoidPrime(list)

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
    def transpose(list):
        return [[list[j][i] for j in range(len(list))] for i in range(len(list[0]))]

    @staticmethod
    def addBias(list,bias):
        temp = []
        for n in list:
            temp.append([n[0]+bias])
        return temp