def clean(files, o):
	out = open(o, 'w')

	first = True

	for f in files:
		lines = open(f, 'r').readlines()

		if not first:
			lines = lines[1:]

		if first:
			first = False

		for line in lines:
			l = line.strip()

			cs = l.split(',')

			ts = []

			if len(cs) > 6:
				ts.append(cs[0])
				ts.append(','.join(cs[1:-4]))
				ts.extend(cs[-4:])

			else:
				ts = cs

			out.write('\t'.join(ts))

clean(['../csv/train_full.csv'], '../tsv/train.tsv')
clean(['../csv/dev_full.csv'], '../tsv/dev.tsv')
clean(['../csv/test_full.csv'], '../tsv/test.tsv')
