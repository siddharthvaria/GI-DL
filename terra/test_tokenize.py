import csv
import subprocess

def tokenize_tweet(text):
    print text
    tokenized = subprocess.check_output(['/proj/nlp/users/terra/gang-intervention/tweet_preprocessing/individual_tweet_tokenize.sh', text])
#    tokenized = tokenized.rstrip().split('\n')[-1]
                   
    print tokenized
    return tokenized

input = u'what: is upU+1F61CU+1F61C'
output = tokenize_tweet(input)

