def csv_to_tsv(files, o=None):
	out = None
	
	if o != None:
		out = open(o, 'w')

	first = True

	for f in files:
		if out == None:
			out = open(f, 'w')
		
		lines = open(f, 'r').readlines()

		a = 0 if first else 1
    
		for line in lines[a:]:
			l = line.strip()

			cs = l.split(',')

			ts = cs
        
			if len(cs) > 8:
				ts.append(cs[0].strip()
				
				for i in range(0, 6):
					ts.append(cs[i].strip())

				ts.append(','.join(cs[6:-1]).strip())

			else:
				ts = cs[:-1]
            
			l = '\t'.join(ts)
        
			out.write(l.lower() + '\n')

def clean_labeled(f, o=None):
	out = open(f, 'w') if o == None else open(o, 'w')

	for line in open(f, 'r').readlines():
		l = line.strip()

		cs = l.split(',')

		ts = []

		if len(cs) == 8:
			pass
		pass
	pass
	
csv_to_tsv(['../csv/All-Tweets-1.csv', '../csv/All-Tweets-2.csv'], 'all.tsv')
