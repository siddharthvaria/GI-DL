a = open('results/significance_testing/CC_aggress_a', 'r')
b = open('results/significance_testing/CC_aggress_b', 'r')

count = 0
total_count = 0
for line_a in a:
	line_b = b.readline()
	if line_a == line_b: count += 1
	total_count += 1

print total_count-count, total_count