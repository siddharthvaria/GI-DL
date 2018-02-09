import csv
import os
import numpy as np

working_dir = os.getcwd()
clf1_out = os.path.join(working_dir, "classifier1.tsv")
clf2_out = os.path.join(working_dir, "classifier2.tsv")
clf3_out = os.path.join(working_dir, "classifier3.tsv")

weights = [1.0,1.0,1.0]

def return_prob(classifier_path):
	all_prob = []
	with open(classifier_path) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			line = np.asarray([float(val) for val in line], dtype=np.float32)
			all_prob.append(line)
	return all_prob


def create_ensemble(clf1_file, clf2_file, clf3_file):
	prob1 = return_prob(clf1_file)
	prob2 = return_prob(clf2_file)
	prob3 = return_prob(clf3_file)
	results = []
	for i in range(0,len(prob1)):
		weighted_sum = weights[0] * prob1[i] + weights[1] * prob2[i] + weights[2] * prob3[i]
		results.append(np.argmax(weighted_sum))
	return results

result = create_ensemble(clf1_out, clf2_out, clf3_out)

predictions_file = os.path.join(working_dir, "ensemble_predictions.txt")
with open(predictions_file, "w") as out_file:
	for line in result:
		out_file.write("{}\n".format(line))
out_file.close()
