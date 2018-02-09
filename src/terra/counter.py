import csv 
import sys

data_file = 'data/data.csv'

reload(sys)
sys.setdefaultencoding("utf-8")


f = open(data_file, 'rU')
reader = csv.DictReader(f)

count = 0

S = ['u+']

for row in reader:
	for s in S:
		if s in row['CONTENT'].lower():
			count+=1

print count