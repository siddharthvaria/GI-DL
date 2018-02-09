import numpy as np
import pickle

'''
Provides an interface for combining word embeddings. Currently only supports
addition, multiplication and averaging, but more functionality will be added
later.
'''

class Embedder:
	def __init__(self, model='paired_vocab.pkl', dim=100, combination='addition'):
		self.model = pickle.load(open(model, 'r'))
		self.dim = dim
		self.method = combination # use addition, multiplication, averaging

		if combination == 'addition':
			self.default = np.zeros(dim)
		elif combination == 'multiplication':
			self.default = np.ones(dim)
		elif combination == 'averaging':
			self.default = np.zeros(dim) # TODO: try using average of all vectors in vocabulary instead
		else:
			self.default = np.zeros(dim) # the real default - can set to different things later

	def ones(self):
		return np.ones(dim)

	def zeros(self):
		return np.zeros(dim)

	def embed(self, text):
		'''
		    Returns a naive combination of all words in the text.
			Takes single words or multi-word texts.
		'''
        
		vectors = []

		text = text.lower()

		empty = True

		for word in text.split():
			if word in self.model:
				empty = False
				break

		if empty:
			return None
        
		for word in text.split():
			if word in self.model:
				vectors.append(self.model[word])

		if len(vectors) == 1:
			return vectors[0]

		return self.combine(vectors)

	def embed_tokens(self, tokens):
		return self.embed(' '.join(tokens))
    
	def combine(self, vectors):
#		print np.shape(vectors)
        
		if len(vectors) == 1:
			return vectors[0]
		
		if self.method == 'addition':
			sm = self.default
			for v in vectors:
				sm = np.add(sm, v)
			return sm

		if self.method == 'multiplication':
			p = vectors[0]

			for v in vectors[1:]:
				p = np.multiply(p, v)
			
			return p

		if self.method == 'averaging':
			sm = self.default
			for v in vectors:
				sm = np.add(sm, v)
			return sm/len(vectors)

		else:
			print 'No such method as ' + self.method
