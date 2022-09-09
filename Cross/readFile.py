class readFile():
    def __init__(self, filePath, validationNumber, validationRange) -> None:
        self.filePath = filePath

        File1 = open(self.filePath, 'r')
        Lines = File1.readlines()
        dataList = []
        validationDataList = []
        outputList = []
        validationOutputList = []
        temp = []

        self.data = dataList

        for i in range(len(Lines)):
            if i % 3 == 1:
                temp = Lines[i].strip('\n').split('  ')
                temp = [float(x) for x in temp]
                dataList.append(temp)
            if i % 3 == 2:
                temp = Lines[i].strip('\n').split(' ')
                temp = [float(x) for x in temp]
                outputList.append(temp)
            temp = []

        for i in range(int(len(dataList) / validationRange)):
            validationDataList.append(dataList.pop(validationNumber * int(len(dataList) / validationRange)))
            validationOutputList.append(outputList.pop(validationNumber * int(len(dataList) / validationRange)))
        self.validationData = validationDataList
        self.validationDesireOutput = validationOutputList
        self.trainingData = dataList
        self.desireOutput = outputList


    def normData(self,list,minVal,maxVal):
        temp = []
        for n in list:
            temp.append((n-minVal)/(maxVal-minVal))
        return temp
