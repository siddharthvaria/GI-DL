import csv
import re

file_name = 'top_communicators_labeled.csv'
out_file = 'aave_stopwords.txt'

#w = open(out_file, 'wb')



f = open(file_name, 'rU')
reader = csv.DictReader(f)

word_counts = {}

for row in reader:
    content = re.sub('RT\s', '', row['CONTENT'])
    tokens = content.lower().split()
    for t in tokens:
    	word_counts[t] = word_counts[t]+1
    	except:
    		word_counts[t] = 1
    
top_100 = dict(sorted(A.iteritems(), key=operator.itemgetter(1), reverse=True)[:100])

for word in top_100.keys():
	print word + ' ' + top_100[word]

f.close()