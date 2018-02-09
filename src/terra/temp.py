import pickle

inf = open('combined_all_reconciled/dev.csv', 'r')
out = open('comparison.txt', 'w')
preds = pickle.load(open('predictions.txt', 'r'))

lines = inf.readlines()

i = 1
j = 0

while i < len(lines) and j < len(preds):
    if len(lines[i].split(',')) != 6:
        print lines[i] + '\t' + str(len(lines[i].split(',')))
        i += 1
        continue

    out.write(lines[i].strip() + '\t' + str(preds[j]) + '\n')
    i += 1
    j += 1
