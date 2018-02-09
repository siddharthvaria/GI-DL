import pickle
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#preds = pickle.load(open('predictions.txt', 'r'))
#out = open('terra_probs.tsv', 'w')
#out.write('Loss\tOther\tAggression\n')

#for pred in preds:
#    p = [float(pred[2][0]), float(pred[2][1]), float(pred[2][2])]
#    s = softmax(p)
#    out.write(str(s[0]) + '\t' + str(s[1]) + '\t' + str(s[2]) + '\n')

def concatenate(f1, f2, outf):
    lines1 = open(f1, 'r').readlines()
    lines2 = open(f2, 'r').readlines()

    out = open(outf, 'w')

    for line in lines1:
        out.write(line)

    for line in lines2:
        out.write(line)

concatenate('Distant_Supervision_dataset/distant_supv2/train.csv', 'nov-new-dataset/train.csv', 'train.csv')
concatenate('Distant_Supervision_dataset/distant_supv2/dev.csv', 'nov-new-dataset/dev.csv', 'dev.csv')
