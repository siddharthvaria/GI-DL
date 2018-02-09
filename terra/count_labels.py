import csv
import sys
sys.path.append('code/tools')
from tools import get_label_sets

f = open('data/_train/train_full.csv', 'rU')
reader = csv.DictReader(f)
aggress, loss = get_label_sets()
loss_count = 0
aggress_count = 0
gen_count = 0
once = False

for row in reader:
	label = row['LABEL'].lower().replace(',', '/')
	found = False
	for a in aggress:
		if a in label and not found: 
			aggress_count += 1
			found = True
			break
	for l in loss:
		if l in label and not found: 
			loss_count += 1
			found = True
			break
	if not found: gen_count += 1

print 'train stats'
print 'aggress_count = '+str(aggress_count)
print 'loss_count = '+str(loss_count)
print 'gen_count = '+str(gen_count)
print 'total count = '+str(loss_count+aggress_count+gen_count)

f = open('data/_dev/dev_full.csv', 'rU')
reader = csv.DictReader(f)
aggress, loss = get_label_sets()
loss_count = 0
aggress_count = 0
gen_count = 0
once = False

for row in reader:
	label = row['LABEL'].lower().replace(',', '/')
	found = False
	for a in aggress:
		if a in label and not found: 
			aggress_count += 1
			found = True
			break
	for l in loss:
		if l in label and not found: 
			loss_count += 1
			found = True
			break
	if not found: gen_count += 1

print 'dev stats'
print 'aggress_count = '+str(aggress_count)
print 'loss_count = '+str(loss_count)
print 'gen_count = '+str(gen_count)
print 'total count = '+str(loss_count+aggress_count+gen_count)

f = open('data/_test/test_full.csv', 'rU')
reader = csv.DictReader(f)
aggress, loss = get_label_sets()
loss_count = 0
aggress_count = 0
gen_count = 0
once = False

for row in reader:
	label = row['LABEL'].lower().replace(',', '/')
	found = False
	for a in aggress:
		if a in label and not found: 
			aggress_count += 1
			found = True
			break
	for l in loss:
		if l in label and not found: 
			loss_count += 1
			found = True
			break
	if not found: gen_count += 1

print 'test stats'
print 'aggress_count = '+str(aggress_count)
print 'loss_count = '+str(loss_count)
print 'gen_count = '+str(gen_count)
print 'total count = '+str(loss_count+aggress_count+gen_count)