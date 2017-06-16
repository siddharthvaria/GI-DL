
# coding: utf-8

# In[53]:

import keras
from gensim.models import Word2Vec
from bible_utils import *
import word2vec_utils as w2v
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Dropout, Activation, Embedding, TimeDistributed
import numpy as np
from style_encoder_utils import *
from time import time


# In[54]:

bible_map = get_bible_map()
num_bible_versions = len(bible_map)
w2v_model = w2v.initialize()
my_verse = get_verse( "John",3,16)
output_map = dict()
for i,version in enumerate( bible_map.keys() ):
    target_array = np.zeros( num_bible_versions )
    target_array[i] = 1
    output_map[version] = target_array


# In[98]:

const_my = json.load( open( 'data/' + 'constitution' + '.json' ) )
taylor   = json.load( open( 'data/' + 'Taylor' +  '.json'))
eminem   = json.load( open( 'data/' + 'Eminem' +  '.json'))
movies   = json.load( open( 'data/' + 'Movies' +  '.json'))
aux_sources = [const_my,taylor,eminem,movies]


# In[100]:

input_max_length = 150
bible_map.keys()
len(w2v.get_unknown_vectors())


# In[106]:

start = time()

style_vects, objectives = get_n_encoded_training_pairs( bible_map, aux_sources,w2v_model, 10000, input_max_length = input_max_length )
end = time()
per_entry = (end - start ) / len(objectives)
print( per_entry )
len(w2v.get_unknown_vectors())


# In[107]:

data_dim = len(style_vects[0][0])
output_size = len(objectives[0])
dropout_level = 0.15

styleInput = Input(shape=(input_max_length,data_dim)) # 150 size 301 word vectors

# define style model
a = LSTM(data_dim-51, return_sequences = True, dropout=dropout_level, recurrent_dropout=dropout_level, name="style_1")
b = LSTM(data_dim-151, return_sequences = True, dropout=dropout_level, recurrent_dropout=dropout_level, name="style_2")
c = LSTM(data_dim-201, return_sequences = False, dropout=dropout_level, recurrent_dropout=dropout_level, name="style_3")
d = Dropout(0.7,name='style_throwaway_0')
e = Dense(output_size, activation='softmax',use_bias=False,name='style_throwaway_a')
a.trainable = True
b.trainable = True
c.trainable = True

style = e(d(c(b(a(styleInput)))))
# load style model's weights from pre-training
model = Model(styleInput, style)

rms = keras.optimizers.RMSprop(lr = 0.0006)
model.compile(loss = 'categorical_crossentropy',
              optimizer=rms,
              metrics=['accuracy']
              )


# In[104]:

model.load_weights('style_weights.h5',by_name = True )
model.fit(style_vects, objectives,
          batch_size=250, epochs=5,
          validation_split=0.05
         )
model.save('style_weights.h5')


# In[105]:

test_size = len(style_vects)//20
test_style_vects, test_objectives = get_n_encoded_training_pairs( bible_map, aux_sources,w2v_model, test_size, input_max_length = input_max_length )
score = model.evaluate(test_style_vects, test_objectives, batch_size=64)
print("\nModel Accuracy: %.2f%%" % (score[1]*100))


# In[73]:

model.summary()
#model.save('style_weights.h5')


# In[ ]:

objectives[0]


# In[5]:

import pickle 
unknown_path  = 'unknown_words_stored.pkl'

def get_unknown_vectors():
    with open(unknown_path, 'rb') as f:
        unknown_vectors = pickle.load(f)
    return( unknown_vectors )


# In[7]:

import pickle 
unknown_path  = 'unknown_words_stored.pkl'


# In[9]:

w2v.save_unknown_vectors({"__BLANK__":np.array([0]*300+[1])})


# In[17]:

len(w2v.get_unknown_vectors())


# In[38]:

get_verse("Genesis",1,1)


# In[39]:

get_verse("Revelation",2,3)


# In[48]:

get_verse("Luke",2,5)


# In[42]:

get_verse("Exodus",2,4)


# In[46]:

get_verse("2 Corinthians",2,6)


# In[108]:

sum(objectives)


# In[109]:

len(objectives[0])


# In[ ]:



