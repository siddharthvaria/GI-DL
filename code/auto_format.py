import pickle

lines = open('../tsv/all.tsv', 'r').readlines()[1:]
out = open('../tsv/auto.tsv', 'w')

for line in lines:
#	print line
	l = line.split('\t')[6].lower().strip()
	sentence = l.split()

        outline = ''

        for c in l:
                outline = outline + c + ' '

        outline = outline + '|'

        for c in l:
                outline = outline + ' ' + c
        
        out.write(outline + '\n')
