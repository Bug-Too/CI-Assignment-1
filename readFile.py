class readFile():
    def __init__(self,filePath,startLine) -> None:
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
                temp = self.normData(temp,minVal,maxVal)
                desireOutput.append(temp.pop()) 
                data.append(temp)
        self.data = data
        self.desireOutput = desireOutput
        self.minVal = minVal
        self.maxVal = maxVal



    def normData(list,minVal,maxVal):
        temp =[]
        for n in list:
            temp.append((n-minVal)/(maxVal-minVal))
        return temp