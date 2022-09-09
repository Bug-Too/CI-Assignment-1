class readFile():
    def __init__(self, filePath, startLine, validationNumber, validationRange) -> None:
        self.filePath = filePath
        self.startLine = startLine

        File1 = open(self.filePath, 'r')
        Lines = File1.readlines()
        count = 0
        startLine = self.startLine
        data = []
        temp = []
        listOfAllData = []
        desireOutput = []
        for line in Lines:
            count += 1
            if count > startLine - 1:
                temp = line.strip('\n').split('\t')
                temp = [float(x) for x in temp]
                listOfAllData.extend(temp)
        temp = []
        count = 0
        minVal = min(listOfAllData)
        maxVal = max(listOfAllData)
        for line in Lines:
            count += 1
            if count > startLine - 1:
                temp = line.strip('\n').split('\t')
                temp = [float(x) for x in temp]
                temp = self.normData(temp, minVal, maxVal)
                desireOutput.append([temp.pop()])
                data.append(temp)
        self.data = data
        self.output = desireOutput
        self.validationData = []
        self.trainingData = data
        self.desireOutput = desireOutput
        self.validationDesireOutput = []
        for i in range(int(len(self.trainingData) / validationRange)):
            self.validationData.append(self.trainingData.pop(validationNumber * int(len(self.trainingData) / validationRange)))
            self.validationDesireOutput.append(self.desireOutput.pop(validationNumber * int(len(self.trainingData) / validationRange)))

        self.minVal = minVal
        self.maxVal = maxVal

    def normData(self, list, minVal, maxVal):
        temp = []
        for n in list:
            temp.append((n - minVal) / (maxVal - minVal))
        return temp
