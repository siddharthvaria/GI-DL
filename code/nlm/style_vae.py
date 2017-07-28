# coding: utf-8

import keras
from gensim.models import Word2Vec
from bible_utils import *
import word2vec_utils as w2v
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Dropout, Activation, Embedding, TimeDistributed, RepeatVector, Lambda
from keras import backend as K
from keras import metrics
import numpy as np
from style_encoder_utils import *
from time import time
import pickle
from keras_vae_example import sampling, CustomVariationalLayer

# get data
bible_map = get_bible_map()
num_bible_versions = len(bible_map)
w2v_model = w2v.initialize()
split_verses = 'verses_in_5.pkl'
split = 0
my_verse = get_verse( "John",3,16)
output_map = dict()
for i,version in enumerate( bible_map.keys() ):
    target_array = np.zeros( num_bible_versions )
    target_array[i] = 1
    output_map[version] = target_array

# initialize constants
const_my = json.load( open( 'data/' + 'constitution' + '.json' ) )
input_max_length = 50
pretrain = False#True

start = time()

style_vects, objectives = get_n_encoded_training_pairs( bible_map, [const_my],w2v_model, 2500, input_max_length = input_max_length )
end = time()
per_entry = (end - start ) / len(objectives)
print( per_entry )
len(w2v.get_unknown_vectors())

# model constants
data_dim = len(style_vects[0][0])
batch_size = 100
output_size = len(objectives[0])
dropout_level = 0.15
latent_dim = 8
intermediate_dim = 256
epochs = 16
epsilon_std = 1.0

# model definition
#model = Sequential()
#model.add(LSTM(data_dim,input_shape=(input_max_length,data_dim),return_sequences=True, 
#              recurrent_dropout=dropout_level, dropout=dropout_level,name='style_1'))
#model.add(LSTM(data_dim-101,return_sequences=True, recurrent_dropout=dropout_level, dropout=dropout_level,name='style_2'))
#model.add(LSTM(data_dim-201,return_sequences=False, recurrent_dropout=dropout_level, dropout=dropout_level,name='style_3'))

#model.add(Dropout(0.4,name='style_throwaway_0'))
#model.add(Dense(output_size, activation='softmax',use_bias=False,name='style_throwaway_1'))
#rms = keras.optimizers.RMSprop(lr = 0.001)
#model.compile(loss = 'categorical_crossentropy',
#              optimizer=rms,
#              metrics=['accuracy']
#              )

x = Input(batch_shape=(batch_size, input_max_length, data_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mu = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mu, z_log_sigma])

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mu = Dense(output_size, activation='sigmoid')
#decoder_mu = Dense(data_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mu = decoder_mu(h_decoded)

y = CustomVariationalLayer()([x, x_decoded_mu])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

# pretrain vae
if pretrain:
    # set up data
    print 'Pretraining VAE...'
    
    x_train = []

    vs = pickle.load(open(split_verses, 'r'))[split]

    for v in vs:
        verse = get_verse(v[0], v[1], v[2])

        skip = False
        
        for k in verse.keys():
            sentence = verse[k]

            vectors = []
            
            for v in w2v.vectorize(w2v.initialize(), sentence,
                                   pad_length=input_max_length):
                vectors.append(v)

            if not len(vectors) > input_max_length:
                x_train.append(vectors)

    print np.shape(np.array(x_train))

    vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size)

    print 'Pretraining done.'

print 'Setting up encoder...'
    
# encoder
encoder = Model(x, z_mu)

# classifier
#clf_input = Input(shape=(latent_dim,))
clf = Dense(data_dim, activation='softmax',use_bias=False,name='style_throwaway_1')(x_decoded_mu)
model = Model(clf_input, clf)

# train
#model.load_weights('style_weights.h5')
model.fit(style_vects, objectives,
          batch_size=128, epochs=3,
          validation_split=0.05
         )
model.save('style_weights.h5')

# evaluate
test_size = len(style_vects)//20
test_style_vects, test_objectives = get_n_encoded_training_pairs( bible_map, [const_my],w2v_model, test_size, input_max_length = input_max_length )
score = model.evaluate(test_style_vects, test_objectives, batch_size=64)
print("\nModel Accuracy: %.2f%%" % (score[1]*100))

model.summary()
#model.save('style_weights.h5')

data_dim

# unknown vectors
unknown_path  = 'unknown_words_stored.pkl'

def get_unknown_vectors():
    with open(unknown_path, 'rb') as f:
        unknown_vectors = pickle.load(f)
    return( unknown_vectors )

unknown_path  = 'unknown_words_stored.pkl'

w2v.save_unknown_vectors({"__BLANK__":np.array([0]*300+[1])})

len(w2v.get_unknown_vectors())
