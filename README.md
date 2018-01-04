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

### Data formats
* Labeled .csv files are assumed to have the following columns: 'text', 'label' and 'tweet_id'.
* Unlabeled .csv files are assumed to have the following columns: 'text' and 'tweet_id'.
* The order of the columns does not matter as long as there is a header line in the .csv files.

### To learn about the command line arguments
* python preprocess_tweets.py --help
* python train.py --help
* python test.py --help

### To preprocess the datasets
* Before you can train the classifier, you need to preprocess the various data files. Preprocessing will build the vocabulary and also preprocess the tweets. Run the following command to preprocess the data files:
    * python preprocess_tweets.py -tr path/to/labeled/train/file -val path/to/labeled/validation/file -tst path/to/labeled/test/file -ofd output/directory/where/you/want/preprocessed/files/to/be/saved --unld_tr path/to/unlabeled/train/file --unld_val path/to/unlabeled/validation/file --word_level True
* The last flag in previous command, --word_level can be used to preprocess the tweets at either character level or word level. By default, the tweets are processed at character level and if word_level is True, then tweets are processed at word level.
* The preprocessing script will create new files in the output directory mentioned in the above command. There will be one file corresponding to each of train, validation, test and unlabeled files.
* These new files will serve as input files to either the classifier or language model.

### To train the classifier
* To train the classifier with either CNN or LSTM architecture:
    * python -u train.py -sdir ../where/you/want/to/save/the/trained/model -md clf/lm -at cnn/lstm -tr ../path/to/labeled/training/data/file -val ../path/to/labeled/validation/data/file -tst ../path/to/labeled/test/data/file -dict ../path/to/dictionary/file
* To restore training of classifier from previously saved model, use the -tm flag and point it to the trained model file.

### To train the Language model
* To train a language model with LSTM architecture:
    * python -u train.py -md lm -at lstm -sdir ../where/you/want/to/save/the/trained/model -dict ../path/to/dictionary/file -unld_tr ../path/to/unlabeled/training/data/file -unld_val ../path/to/unlabeled/validation/data/file

### To get test predictions
* To get predictions on a test set using a trained model, use the following command:
    * python test.py -tst path/to/test/file -wl True -tm path/to/trained/model/file
* To process test file at character level, drop the -wl in the above command.
