import random

class initalNetwork():
    
    def __init__(self,layers) -> None:
        self.layer = layers
        self.weight = self.initWeight()
        self.node = self.initActivation()

    def initWeight(self):
        weights = []
        tempRand = []
        tempWeight = []
        for i in range(len(self.layers)-1):
            for j in range(self.layers[i]):
                for k in range(self.layers[i+1]):
                    tempRand.append(random.random())
                tempWeight.append(tempRand)
                tempRand = []
            weights.append(tempWeight)
            tempWeight = []
        return weights
    
    def initActivation(self):
        initVal = 1
        initActivationVal = []
        temp = [] 
        for layer in self.layers:
            for i in range(layer):
                temp.append(initVal)
            initActivationVal.append(temp)
            temp = []
        return initActivationVal
    
