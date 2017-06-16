import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, RepeatVector, BatchNormalization, concatenate
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
from os.path import isfile

from gensim.models import Word2Vec

import os
import json
import random
import numpy as np
import pickle

import bible_constants
from bible_utils import get_verse, get_bible_map
from word2vec_utils import vectorize, initialize, reset_global_unknown_vectors, save_global_unknown_vectors

K.set_learning_phase(1)
K._LEARNING_PHASE = K.constant(1)

outf = 'content_weights.h5'
boutf = 'content_backup_weights.h5'
latent_dim = bible_constants.EMBEDDING_DIMENSION
dim = bible_constants.EMBEDDING_DIMENSION
verse_length = bible_constants.MAX_VERSE_LENGTH
null = bible_constants.NULL_VECTOR

split_verses = 'verses_in_5.pkl'

w2v = initialize()

def sched(epoch):
    if epoch < 2:
        return 0.001
    elif epoch < 5:
        return 0.0007
    else:
        return 0.0005

def get_sentences(split, model=w2v):
    '''
        Helper function
    '''
    sentences = []
    verses = []

    reset_global_unknown_vectors()
    bible_map = get_bible_map()

    print bible_map.keys()

    keys = ['BBE', 'ESV', 'KJV', 'MSG', 'NIV', 'NLT', 'YLT']

    # obtain text verses
#    for book in bible_map['KJV']:
#        for chapter in bible_map['KJV'][book]:
#            for verse in bible_map['KJV'][book][chapter]:
#                if random.random() < (1. - percent):
#                    continue
#                try:
#                    v = get_verse(book, chapter, verse)
#                    verses.append(v)
#                except:
#                    continue

    vs = pickle.load(open(split_verses, 'r'))[split]

    for v in vs:
        verses.append(get_verse(v[0], v[1], v[2]))
                
    print len(verses)

    # obtain word vector sentences
    for verse in verses:
        
        vsentences = []

        skip = False
        
        for key in keys:
            # preprocessing
            sentence = verse[key]

            # translate text into embeddings
            vectors = []
            for v in vectorize(model, sentence, pad_length = verse_length):
                vectors.append(v)

            if len(vectors) > verse_length:
                skip = True

            vsentences.append(vectors)

        if skip:
            continue

        sentences.append(np.array(vsentences))

    sentences = np.array(sentences)
    save_global_unknown_vectors()
    return sentences

#define model

input = Input((verse_length, dim))

# names content_1, content_2, content_3
#content_encoder = LSTM(latent_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, name='content_1')(input)
#content_encoder = BatchNormalization()(content_encoder)
#content_encoder = LSTM(latent_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, name='content_2')(content_encoder)
#content_encoder = BatchNormalization()(content_encoder)
#content_encoder = LSTM(latent_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, name='content_3')(content_encoder)
#content_encoder = BatchNormalization()(content_encoder)

content_encoder = Bidirectional(LSTM(latent_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, name='contet_bi_1'))(input)
content_encoder = LSTM(latent_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, name='content_2')(content_encoder)
content_encoder = LSTM(latent_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, name='content_3')(content_encoder)
content_encoder = LSTM(latent_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, name='content_4')(content_encoder)
content_encoder = BatchNormalization()(content_encoder)

#content_encoder = RepeatVector(verse_length)(content_encoder)
BBE_dec = LSTM(latent_dim, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(content_encoder)

ESV_dec = LSTM(latent_dim, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(content_encoder)

KJV_dec = LSTM(latent_dim, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(content_encoder)

MSG_dec = LSTM(latent_dim, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(content_encoder)

NIV_dec = LSTM(latent_dim, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(content_encoder)

NLT_dec = LSTM(latent_dim, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(content_encoder)

YLT_dec = LSTM(latent_dim, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(content_encoder)


pretrainer = Model(inputs=[input],
                   outputs=[BBE_dec, ESV_dec, KJV_dec, MSG_dec, NIV_dec, NLT_dec, YLT_dec])

pretrainer.load_weights("content_backup_weights.h5")

#rmsprop = RMSprop(lr=0.0007, rho=0.9, epsilon=1e-08, decay=0.0)
pretrainer.compile(optimizer='rmsprop', loss='cosine_proximity',
                   loss_weights=[1., 1., 1., 1., 1., 1., 1.], metrics=['accuracy'])

checkpointer = ModelCheckpoint(outf, verbose=1, monitor='val_loss', save_best_only=True)
lrs = LearningRateScheduler(sched)

# get data
# 1st
#sentences = get_sentences(0)
#
#X = []
#Y = [[], [], [], [], [], [], []]
#
#for i in range(len(sentences)):
#    if len(sentences[i]) < 7:
#        continue
#    
#    X.append(sentences[i][random.randint(0, 6)])
#
#    for j in range(7):
#        Y[j].append(sentences[i][j])
#
#for j in range(7):
#    Y[j] = np.array(Y[j])
#        
#X = np.array(X)
#
#pretrainer.fit(X, Y, epochs=8, batch_size=128, validation_split=0.05, callbacks=[checkpointer, lrs])
#
# 2nd
#if isfile(outf):
#    pretrainer.load_weights(outf)
#pretrainer.save(boutf)
#sentences = get_sentences(1)
#
#X = []
#Y = [[], [], [], [], [], [], []]
#
#for i in range(len(sentences)):
#    if len(sentences[i]) < 7:
#        continue
#    
#    X.append(sentences[i][random.randint(0, 6)])
#
#    for j in range(7):
#        Y[j].append(sentences[i][j])
#
#for j in range(7):
#    Y[j] = np.array(Y[j])
#        
#X = np.array(X)
#
#pretrainer.fit(X, Y, epochs=8, batch_size=128, validation_split=0.05, callbacks=[checkpointer, lrs])
#
## 3rd
#if isfile(outf):
#    pretrainer.load_weights(outf)
#pretrainer.save(boutf)
#sentences = get_sentences(2)
#
#
#for i in range(len(sentences)):
#    if len(sentences[i]) < 7:
#        continue
#    
#    X.append(sentences[i][random.randint(0, 6)])
#
#    for j in range(7):
#        Y[j].append(sentences[i][j])
#
#for j in range(7):
#    Y[j] = np.array(Y[j])
#        
#X = np.array(X)
#
#pretrainer.fit(X, Y, epochs=8, batch_size=128, validation_split=0.05, callbacks=[checkpointer, lrs])
#
## 4th
#if isfile(outf):
#    pretrainer.load_weights(outf)
#pretrainer.save(boutf)
sentences = get_sentences(3)

X = []
Y = [[], [], [], [], [], [], []]

for i in range(len(sentences)):
    if len(sentences[i]) < 7:
        continue
    
    X.append(sentences[i][random.randint(0, 6)])

    for j in range(7):
        Y[j].append(sentences[i][j])

for j in range(7):
    Y[j] = np.array(Y[j])
        
X = np.array(X)

pretrainer.fit(X, Y, epochs=8, batch_size=128, validation_split=0.05, callbacks=[checkpointer, lrs])

# 5th
if isfile(outf):
    pretrainer.load_weights(outf)
pretrainer.save(boutf)
sentences = get_sentences(4)

X = []
Y = [[], [], [], [], [], [], []]

for i in range(len(sentences)):
    if len(sentences[i]) < 7:
        continue
    
    X.append(sentences[i][random.randint(0, 6)])

    for j in range(7):
        Y[j].append(sentences[i][j])

for j in range(7):
    Y[j] = np.array(Y[j])
        
X = np.array(X)

pretrainer.fit(X, Y, epochs=8, batch_size=128, validation_split=0.05, callbacks=[checkpointer, lrs])

pretrainer.save(boutf)
