import sys
sys.path.append('code/tools')
from tools import get_label_sets, get_other_labels
import csv

data_file = 'data/preprocessed/_test/test_full.csv'
out_file = 'data/preprocessed/_test/test_aggress_loss.csv'
f = open(data_file, 'rU')
reader = csv.DictReader(f)

aggress, loss = get_label_sets()
aggress.extend(loss)

w = open(out_file, 'wb')
writer = csv.DictWriter(w, ['AUTHOR', 'CONTENT', 'LABEL', 'DATE', 'URL', 'DESC'])

writer.writeheader()
for row in reader:
	label = row['LABEL'].lower()
	for l in aggress:
		if l in label:
			description = row['DESC']
			d = {'AUTHOR': row['AUTHOR'], 'CONTENT': row['CONTENT'], 'LABEL': label, 'DATE': row['DATE'], 'URL': row['URL'], 'DESC': description}
			writer.writerow(d)
			break

f.close()
w.close()
