import classifier

from preprocess import extract_data

timesteps = 140
clf = classifier.Classifier(timesteps)

train_file = '../../tsv/train.tsv'
test_file = '../../tsv/test.tsv'

X_train, y_train = extract_data(train_file, timesteps)
X_test, y_test = extract_data(test_file, timesteps)

clf.fit_classif(X_train, y_train)
clf.evaluate(X_test, y_test)
