import sys
sys.path.append('code/tools')
from tools import get_label_sets, get_other_labels, tweet_preprocessing
import csv
import re

data_file = 'data/_train/train_full.csv'
out_file = 'data/_train/train_full_no_rip.csv'
f = open(data_file, 'rU')
reader = csv.DictReader(f)

aggress, loss = get_label_sets()

w = open(out_file, 'wb')
writer = csv.DictWriter(w, ['AUTHOR', 'CONTENT', 'LABEL', 'DATE', 'URL', 'DESC'])

writer.writeheader()
for row in reader:
	label = row['LABEL'].lower()
	d = {'AUTHOR': row['AUTHOR'], 'CONTENT': row['CONTENT'], 'LABEL': label, 'DATE': row['DATE'], 'URL': row['URL'], 'DESC': row['DESC']}

	tokens = tweet_preprocessing(row['CONTENT'].lower())
	good_tokens = []
	for t in tokens: 
		if any(c.isalpha() for c in t) and not re.match('(r|b).?i.?p.?', t) and t != 'USER_HANDLE': good_tokens.append(t)
	if len(good_tokens) > 0: writer.writerow(d)

f.close()
w.close()
