from datetime import datetime
fileNum = open('../../all-tweets-output.txt')
fileCSV = open('../data/conversations/All-Tweets.csv')
line = fileCSV.readline()
#print line.strip()+',prediction'
lines = fileCSV.readlines()
line = fileNum.readline()
nums = line[1:-2].split(', ')

data = []

for i in range(len(lines)):
    data.append([nums[i], datetime.strptime(str(lines[i].split(',')[4]), '%m/%d/%Y %H:%M')] )

data.sort(key=lambda x: x[1])
for i in range(len(lines)):
    print str(data[i][1])+' '+data[i][0]
