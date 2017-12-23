# GI-DL
Deep learning for gang violence prediction

### List of important python files under src directory

1. train.py : implementation of 3-way classifier with the option to pre-train a language model or classifier.
2. Seq2Seq.py : implementation of 3-way classifier with the option to pre-train a sequence to sequence model.
3. preprocess_tweets.py : preprocesses the tweets and generates a pickled file and input files that can be input to above classifiers
4. keras_impl/models.py : This file defines the architecture of different models like the language model and sequence to sequence autoencoders
5. TweetReader2.py : used internally by preprocess_tweets.py
6. CascadeClassifier.py : implementation of cascade classifier with the option to pre-train a language model.
7. test.py : load the trained model and get the predictions (probabilities and predicted classes)

## To learn about the command line arguments
* python preprocess_tweets.py --help
* python train.py --help
* python test.py --help

## To train and test the classifier


* To train the classifier with either CNN or LSTM architecture:
..* python -u train.py -sdir ../where/you/want/to/save/the/trained/model -md clf/lm -at cnn/lstm -tr ../path/to/labeled/training/data/file -val ../path/to/labeled/validation/data/file -tst ../path/to/labeled/test/data/file -dict ../path/to/dictionary/file
* To train a language model with LSTM architecture:
..* python -u train.py -md lm -at lstm -sdir ../where/you/want/to/save/the/trained/model -dict ../path/to/dictionary/file -unld_tr ../path/to/unlabeled/training/data/file -unld_val ../path/to/unlabeled/validation/data/file
* To restore training of classifier from previously saved model, use the -tm flag and point it to the trained model file path.
