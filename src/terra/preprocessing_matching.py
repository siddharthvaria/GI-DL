
file_name1 = 'results/dev_full_preprocessing.txt'
file_name2 = 'results/dev_terra_preprocessing.txt'

file1 = open(file_name1, 'r')
file2 = open(file_name2, 'r')

for line1 in file1:
	line1 = line1.strip()
	line2 = file2.readline().strip()

	#in the case where the two preprocessing methods don't match
	if line1 != line2:
		print 'mismatched:'
		print line1
		print line2
		t1 = line1.split(' ')
		t2 = line2.split(' ')
		print 'mismatched on tokens: '+str([t1[i] for i in range(0, len(t1)) if i>len(t2) or t1[i] != t2[i]])
		print
	else:
		print 'matched: '
		print line1
		print line2

file1.close()
file2.close()