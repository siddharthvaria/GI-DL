from classification import classify
import sys
sys.path.append('code/tools')
from features import train_tagger

#note: feats = 0: unigrams, 1: bigrams, 2: pos_tag feats, 3: description, 4: emotion feats, 5: k (num_feats)
#note: results = 0: loss precision, 1: loss recall, 2: loss f1, 3: aggress precision, 4: aggress recall, 5: agress f1

if len(sys.argv)> 1 and sys.argv[1] == 'BCS_aggress':
	train_file = 'data/classification/_train/train_aggress_loss.csv'
	dev_file = 'data/classification/_test/test_aggress_loss.csv'
	model = 'svm'
	label = 'aggress'
	feats = []
	C = 
	loss = 

if len(sys.argv)> 1 and sys.argv[1] == 'BCS_loss': 
	train_file = 'data/classification/_train/train_aggress_loss.csv'
	dev_file = 'data/classification/_test/test_aggress_loss.csv'
	model = 'svm'
	label = 'loss'
	feats = []
	C = 
	loss = 

if len(sys.argv)> 1 and sys.argv[1] == 'TCF_aggress': 
	train_file = 'data/_train/train_full.csv'
	dev_file = 'data/_test/test_full.csv'
	model = 'svm'
	label = 'aggress'
	feats = []
	C = 
	loss = 

if len(sys.argv)> 1 and sys.argv[1] == 'TCF_aggress_loss': 
	train_file = 'data/_train/train_full.csv'
	dev_file = 'data/_test/test_full.csv'
	model = 'svm'
	label = 'loss_aggress'
	feats = []
	C = 
	loss = 

if len(sys.argv)> 1 and sys.argv[1] == 'TCF_loss': 
	train_file = 'data/_train/train_full.csv'
	dev_file = 'data/_test/test_full.csv'
	model = 'svm'
	label = 'loss'
	feats = []
	C = 
	loss = 

pos_tagger = None
if feats[2] == 'u' or feats[2] == 'b':
	pos_tagger = train_tagger()

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)

#output results
print
print 'Results'
print 
print 'sought precision: '+str(results[0])
print 'sought recall: '+str(results[1])
print 'sought f-score: '+str(results[2])
print
print 'nsought precision: '+str(results[3])
print 'nsought recall: '+str(results[4])
print 'nsought f-score: '+str(results[5])