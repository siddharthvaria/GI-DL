input_file = 'data/emotion/emoji-data.txt'
output_file = 'data/emotion/emoji_lex.txt'

inf = open(input_file, 'r')
outf = open(output_file, 'w')

for line in inf:
	line = line.replace(') ', ' ;\t')
	tokens = line.split(' ;\t')
	if len(tokens) > 3 :
		out_line = tokens[-2].split('(')[1].lower()+' : '+tokens[-1].lower()
		outf.write(out_line+'\n')

inf.close()
outf.close()
