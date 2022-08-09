import math
import random

def sigmoid(x):
    temp = []
    for v in x:
        temp.append(1 / (1 + math.exp(-v)))
    return temp

def sigmoidPrime(x):
    temp = []
    for v in x:
        temp.append((1 / (1 + math.exp(-v)))*(1-(1 / (1 + math.exp(-v)))))
    return temp


def initWeight(layers):
    weights = []
    tempRand = []
    tempWeight = []
    for i in range(len(layers)-1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                print(i,j,k) #for debug
                tempRand.append(random.random())
            tempWeight.append(tempRand)
            tempRand = []
        weights.append(tempWeight)
        tempWeight = []
    return weights

def multiplyMatrix(X,Y):
    return [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]

def readFile():
    File1 = open('/home/pooh/Documents/CI/HW1/CI-Assignment-1/Flood_dataset.txt', 'r')
    Lines = File1.readlines()
    count = 0
    startLine = 3
    data = []
    temp = []
    listOfAllData = []
    desireOutput = []
    for line in Lines:
        count += 1
        if count > startLine-1 :
            temp = line.strip('\n').split('\t')
            temp = [float(x) for x in temp]
            listOfAllData.extend(temp)
    temp = []
    count = 0
    minVal = min(listOfAllData)
    maxVal = max(listOfAllData)
    for line in Lines:
        count += 1
        if count > startLine-1 :
            temp = line.strip('\n').split('\t')
            temp = [float(x) for x in temp]
            temp = normData(temp,minVal,maxVal)
            desireOutput.append(temp.pop()) 
            data.append(temp)
    return data,desireOutput,minVal,maxVal

def normData(list,minVal,maxVal):
    temp =[]
    for n in list:
        temp.append((n-minVal)/(maxVal-minVal))
    return temp

layers = [8,5,3,2,1]
bias = 1
weights = initWeight(layers)
data, desireOutput, minVal, maxVal = readFile()

def initActivation(layers):
    initVal = 1
    initActivationVal = []
    temp = [] 
    for layer in layers:
        for i in range(layer):
            temp.append(initVal)
        initActivationVal.append(temp)
        temp = []

    return initActivationVal


activationVal = initActivation(layers)
VVal = initActivation(layers)



# for debug.
# print(weights)
# print(data[0], len(data))
# print(desireOutput, len(desireOutput))
# print(activationVal)
# print(minVal,maxVal)
print(activationVal)




# !!! Pass by ref func
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

    
    # for i in range(len(layers)-1):
    #     for j in range(layers(i+1)):
    #         activationVal[i+1][j] = sigmoid()
    # return 

def backWard():

    return 0


def findSSEAvg(sumError,N):
    return sumError/N




epsilon = 0.01
maxEpoch = 5000
t = 0
sumSquaredErrorAvg = 0



# while sumSquaredErrorAvg > epsilon and t < maxEpoch:
#     t += 1
#     for i in range(len(data)):
#         sumSquaredErrorAvg += feedForward(data,bias)
#         backWard()
#     sumSquaredErrorAvg = sumSquaredErrorAvg / len(data)

