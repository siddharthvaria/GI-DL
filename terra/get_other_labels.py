from tools import get_label_sets
import csv

aggress, loss = get_label_sets()
files = ['data/_train/train_full.csv', 'data/_dev/dev_full.csv', 'data/_test/test_full.csv']
aggress.extend(loss)

others = {}
for f in files:
	reader = csv.DictReader(open(f, 'rb'))
	for row in reader: 
		found = False
		label = row['LABEL'].lower()
		for l in aggress:
			if l in label.lower():
				found = True
		if not found:
			others[label] = 1

print others.keys()