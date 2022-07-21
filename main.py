File1 = open('Flood_dataset.txt', 'r')
Lines = File1.readlines()
count = 0
data = []
for line in Lines:
    count += 1
    if count > 2 :
        data.append(line.strip('\n').split('\t'))
print(data)
print(count)
