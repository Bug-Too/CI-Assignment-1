from audioop import bias
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0,x)

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

def readFile():
    File1 = open('/home/pooh/Documents/CI/HW1/CI-Assignment-1/Flood_dataset.txt', 'r')
    Lines = File1.readlines()
    count = 0
    startLine = 3
    data = []
    temp = []
    desireOutput = []
    for line in Lines:
        count += 1
        if count > startLine-1 :
            temp = line.strip('\n').split('\t')
            desireOutput.append(temp.pop()) 
            data.append(temp)
    return data,desireOutput

layers = [8,5,3,2,1]
bias = 1
weights = initWeight(layers)
data, desireOutput = readFile()
initActivationVal = initActivation(layers)
print(weights)
print(data, len(data))
print(desireOutput, len(desireOutput))
print(initActivationVal)