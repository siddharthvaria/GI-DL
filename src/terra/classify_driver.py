from classification import classify
import sys
sys.path.append('code/tools')
from features import train_tagger
import pickle

#note: feats = 0: unigrams, 1: bigrams, 2: pos_tag feats, 3: description, 4: emotion feats, 5: k (num_feats)
#note: results = 0: loss precision, 1: loss recall, 2: loss f1, 3: aggress precision, 4: aggress recall, 5: agress f1

def drive():
#    train_file = 'data/classification/_train/train_full.csv'
#    dev_file = 'data/classification/_dev/dev_full.csv'

#    train_file = "nov-new-dataset/train.csv"
#    train_file = 'train.csv'
    train_file = 'distant_train.csv'
    
#    dev_file = "nov-new-dataset/dev.csv"
#    dev_file = 'dev.csv'
#    dev_file = 'add.csv'
    #dev_file = 'data/preprocessed/arrogant_bubba.csv'
    dev_file = "nov-new-dataset/test.csv"

    model = 'svm'
    label = 'aggress'
    feats = [1, 1, 'n', 0, 'min_max/all', 1300]
    C=0.3 # original: C=0.3, modified for experimentation purposes on
           # distantly labeled dataset
    loss = 'squared_hinge'

    print 'Training on ' + train_file + ', testing on ' + dev_file

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
    print
    print 'sought precision: '+str(results[6])
    print 'sought recall: '+str(results[7])
    print 'sought f-score: '+str(results[8])

    pickle.dump(results[-1], open('predictions.txt', 'w'))

    return results

drive()
