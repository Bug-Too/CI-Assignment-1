File1 = open('Flood/cross.pat', 'r')
Lines = File1.readlines()
dataList = []
validationDataList = []
outputList = []
validationOutputList = []
temp = []
validationNumber = 9
validationRange = 10

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
    validationOutputList.append(dataList.pop(validationNumber * int(len(dataList) / validationRange)))
print(validationDataList, '\n', outputList)
print(dataList)
