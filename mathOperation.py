import math

class mathOperation():
    @staticmethod
    def sigmoid(x):
        temp = []
        for v in x:
            temp.append(1 / (1 + math.exp(-v)))
        return temp

    @staticmethod
    def sigmoidPrime(x):
        temp = []
        for v in x:
            temp.append((1 / (1 + math.exp(-v)))*(1-(1 / (1 + math.exp(-v)))))
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
