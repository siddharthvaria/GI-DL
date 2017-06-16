import keras
import numpy as np

import classifier

from preprocess import extract_data

timesteps = 140
clf = classifier.Classifier(timesteps)

train_file = '../../tsv/train.tsv'
#train_file = '../../tsv/all_labeled.tsv'
test_file = '../../tsv/dev.tsv'
#test_file = '../../tsv/train.tsv'
unlabeled_file = '../../tsv/all.tsv'

#X_train, y_train = extract_data(train_file, timesteps, label_split=-1, cap=5000, read='irregular')
X_train, y_train = extract_data(train_file, timesteps, label_split=2)
X_test, y_test = extract_data(test_file, timesteps)
X_auto = extract_data(unlabeled_file, timesteps, use_y=False, text_split=-1, cap=20000)

Y_train = keras.utils.to_categorical(y_train, 3)
Y_test = keras.utils.to_categorical(y_test, 3)

print np.shape(X_train)
print np.shape(Y_train)
#print np.shape(X_auto)

print Y_train.shape[1:]

counts = [0, 0, 0]

# TODO: count and print labels
for y in y_train:
    counts[y] += 1

print counts
print len(y_train)
#exit(0)

#print X_auto

#for i in range(len(X_auto)):
#    x = X_auto[i]
#    
#    for c in x:
#        if c.isalpha() or c[0].isalpha():
#            print x
#            del X_auto[i]
#            i -= 1

#clf.fit_auto(X_auto)

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

clf.fit_classif(X_train, Y_train, validation_data=(X_test, Y_test))#, e=4, b=256)

#print(clf.evaluate(X_test, Y_test))

print 'Testing...'

p = clf.predict(X_test)

for i in range(len(p)):
    print str(p[i]) + ', ' + str(Y_test[i])

print clf.evaluate(X_test, Y_test)
