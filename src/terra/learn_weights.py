from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import numpy as np
import csv

clf = LinearRegression()

X = np.zeros((4194, 9))
y = []

fs = ['char_train_probs.tsv', 'word_train_probs.tsv', 'terra_train_probs.tsv']

for i in range(len(fs)):
    f = fs[i]
    
    lines = open(f, 'r').readlines()[1:]

    for j in range(len(lines)):
        line = lines[j]
        s = line.split('\t')

        X[j][i*3] = float(s[0])
        X[j][i*3 + 1] = float(s[1])
        X[j][i*3 + 2] = float(s[2].strip())

reader = csv.DictReader(open('train.csv', 'r'))

for row in reader:
    l = row['LABEL'].lower()
    y.append(0 if 'loss' in l else 2 if 'aggress' in l else 1)
        
clf = LinearRegression()
clf.fit(X, y)

print 'Coefficient: ' + str(clf.coef_)
print 'Intercept: ' + str(clf.intercept_)

X_test = np.zeros((, 9))

fs = ['char_probs.tsv', 'word_probs.tsv', 'terra_probs.tsv']

for i in range(len(fs)):
    f = fs[i]
    
    lines = open(f, 'r').readlines()[1:]

    for j in range(len(lines)):
        line = lines[j]
        s = line.split('\t')

        X_test[j][i*3] = float(s[0])
        X_test[j][i*3 + 1] = float(s[1])
        X_test[j][i*3 + 2] = float(s[2].strip())

y_pred = clf.predict(X_test)

y_test = []

reader = csv.DictReader(open('dev.csv', 'r'))

for row in reader:
    l = row['LABEL'].lower()
    y.append(0 if 'loss' in l else 2 if 'aggress' in l else 1)

print classification_report(y_test, y_pred)
