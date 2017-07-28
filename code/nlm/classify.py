import keras
import numpy as np

#import classifier
#import vae_classifier as classifier
import cascading_classifier as classifier
from classifier import precision, recall, fmeasure

from preprocess import extract_data
from utils import array_to_text as att, backwards

bidirectional = False
use_embedding = True
use_auto = True
auto_samples = 5000

cw = { 0: 3.,
1: .5,
2: 1.5
}

timesteps = 140
clf = classifier.Classifier(timesteps, bidirectional=bidirectional)

train_file = '../../tsv/unreconciled.tsv'
#train_file = '../../tsv/train.tsv'
#train_file = '../../tsv/all_labeled.tsv'
test_file = '../../tsv/dev.tsv'
#test_file = '../../tsv/train.tsv'
unlabeled_file = '../../tsv/all.tsv'
#unlabeled_file = '../../tsv/fly_girllashea_march_15_to_april_17_2014.tsv'

X_train, y_train = extract_data(train_file, timesteps, label_split=-1)
#X_train, y_train = extract_data(train_file, timesteps, label_split=2)
#X_train, y_train = extract_data(train_file, timesteps, label_split=-1, cap=5000, read='irregular')
X_test, y_test = extract_data(test_file, timesteps)
X_auto = extract_data(unlabeled_file, timesteps, use_y=False, text_split=-1, cap=auto_samples)
#X_auto = extract_data(unlabeled_file, timesteps, use_y=False, text_split=1, cap=auto_samples)

#X_test = X_test[:100]
#y_test = y_test[:100]

Y_train = keras.utils.to_categorical(y_train, 3)
Y_test = keras.utils.to_categorical(y_test, 3)

X_train_b = []

for x in X_train:
    x_b = []

    i = len(x) - 1

    while x[i][0] == 0:
        i -= 1

    for j in range(len(x) - i):
        x_b.append(x[len(x) - i - j - 1])

    while len(x_b) < len(x):
        x_b.append([0, 0])

    X_train_b.append(x_b)

X_train_b = np.array(X_train_b)

print np.shape(X_train)
print np.shape(Y_train)
print np.shape(X_train_b)
#print np.shape(X_auto)

print Y_train.shape[1:]

counts = [0, 0, 0]

# TODO: count and print labels
for y in y_train:
    counts[y] += 1

print counts
print len(y_train)
#exit(0)

counts = [0, 0, 0]

for y in y_test:
    counts[y] += 1

print counts
print len(y_test)

#print X_auto

#for i in range(len(X_auto)):
#    x = X_auto[i]
#    
#    for c in x:
#        if c.isalpha() or c[0].isalpha():
#            print x
#            del X_auto[i]
#            i -= 1

print 'Fitting to unlabeled data...'

X_auto_b = backwards(X_auto)

print np.shape(X_auto_b)

if use_auto:
    if bidirectional:
        clf.fit_auto([X_auto, X_auto_b])
    else:
        clf.fit_auto(X_auto)

#
#print X_auto[0]
#print att(clf.predict([auto[0]]))

## balance data
#cap = min(counts)
#
#print 'Cap: ' + str(cap)
#
#counts = [0, 0, 0]
#
#X = []
#Y = []
#
#for i in range(len(X_train)):
#    if counts[y_train[i]] < cap:
#        counts[y_train[i]] += 1
#        X.append(X_train[i])
#        Y.append(Y_train[i])

#print np.shape(X)
#print np.shape(Y)

print 'Fitting...'

if not bidirectional:
    clf.fit_classif(X_train, Y_train, class_weight=cw)#, e=4, b=256)
else:
    clf.fit_classif([X_train, X_train_b], Y_train, class_weight=cw)#, e=4, b=256)

#print(clf.evaluate(X_test, Y_test))

print 'Testing...'

X_test_b = backwards(X_test)

print np.shape(X_test_b)

p = []

if not bidirectional:
    p = clf.predict(X_test)
else:
    p = clf.predict([X_test, X_test_b])

for i in range(len(p)):
    print str(p[i]) + ', ' + str(Y_test[i])

if bidirectional:
    print clf.evaluate([X_test, X_test_b], Y_test)
else:
    print clf.evaluate(X_test, Y_test)

print 'Class counts: ' + str(counts)

#counts = [0, 0, 0]

#for pred in p:
#    counts[pred] += 1

#print counts

for c in range(3):
    print '[Precision, recall, f-measure] on class ' + str(c) + ': ' + str(precision(p, Y_test, c)) + ', ' + str(recall(p, Y_test, c)) + ', ' + str(fmeasure(p, Y_test, c))
