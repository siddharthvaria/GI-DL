import re

d = {}

#open phrase table
filename = 'data/naive_phrase_table.txt'
f = open(filename, 'rU')

#for each line in phrase table
for line in f:
	word = line.strip().split('\t:\t')
	if len(word) != 3:
		continue
	#store word: (trans, prob) in dict if:
	#not in dict or prob > prob in dict AND
	#word != trans and word containts alphanumeric chars
	if (word[0] not in d.keys() or float(word[2]) > d[word[0]][1]) and (word[0] != word[1] and 
		re.match('^[a-zA-Z0-9\']*$', word[0]) and re.match('^[a-zA-Z0-9\'\s]*$', word[1])):
		print word[0], word[1], word[2]
		d[word[0]] = (word[1], float(word[2]))

outfile = 'data/emotion/lexicon.txt'
w = open(outfile, 'a')

#for each key in dict
for key in d.keys():
	#append to lexicon in 'word : trans' format
	w.write(key+' : '+d[key][0]+'\n')

f.close()
w.close()