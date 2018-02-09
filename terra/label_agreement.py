from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import jaccard_distance
import csv
import tools


#data_file = 'data/gakirah/mar29_apr17_no_emojis.csv'
data_file = 'data/_test/test_full.csv'
f = open(data_file, 'rU')
reader = csv.DictReader(f)
counter = 1

aggress, loss = tools.get_label_sets()

def preprocess(text):
	text = text.strip().lower().replace('?', '').replace('(', '/').replace(')','').replace('/', ' ').replace(',', ' ').split()
	if len(text) > 0:
		return text
	else: return ['']


data1= []
data2 = []
for row in reader:
#	annotator1 = preprocess(row['N'])
#	annotator2 = preprocess(row['L'])
#	annotator3 = preprocess(row['A'])
	annotator1 = preprocess(row['LABEL'])
	annotator2 = preprocess(row['LABEL2'])

	if annotator1[0] in aggress: annotator1 = 'A'
	elif annotator1[0] in loss: annotator1 = 'L'
	else: annotator1 = 'M'

	if annotator2[0] in aggress: annotator2 = 'A'
	elif annotator2[0] in loss: annotator2 = 'L'
	else: annotator2 = 'M'

#	data1.append(('N', counter, annotator1[0]))
#	data1.append(('L', counter, annotator2[0]))
#	data1.append(('A', counter, annotator3[0]))
#	data2.append([set(annotator1), set(annotator2), set(annotator3)])
	data1.append(('1', counter, annotator1))
	data1.append(('2', counter, annotator2))
	data2.append([set(annotator1), set(annotator2)])

	counter += 1

print data1
t = AnnotationTask(data=data1)
print 'avg kappa = '+str(t.kappa())
#print 'multi_kappa = '+str(t.multi_kappa())

total = 0
counter = 0
for t in data2:
#	counter += 3
	counter += 1
	d1 = len(t[0].intersection(t[1])) / float(len(t[0].union(t[1])))
#	d2 = len(t[0].intersection(t[2])) / float(len(t[0].union(t[2])))
#	d3 = len(t[1].intersection(t[2])) / float(len(t[1].union(t[2])))
#	total += float(d1+d2+d3)
	total += float(d1)
print 'avg jaccard distance = '+ str(total/(counter))
