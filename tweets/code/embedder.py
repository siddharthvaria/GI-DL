import numpy as np
from gensim.models import Word2Vec

'''
Provides an interface for combining word embeddings. Currently only supports
addition, multiplication and averaging, but more functionality will be added
later.
'''

class Embedder:
	def __init__(self, model='tweets.model', dim=300, combination='multiplication'):
		self.model = Word2Vec.load(model)
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
		
		for word in text.split():
			v = self.model[word] if word in self.model else self.default
			vectors.append(v)

		if len(vectors) == 1:
			return vectors[0]

		return self.combine(vectors)

	def combine(self, vectors):
		if len(vectors) == 1:
			return vectors[0]
		
		if self.method == 'addition':
			return np.sum(vectors)

		if self.method == 'multiplication':
			p = vectors[0]

			for v in vectors[1:]:
				p = np.multiply(p, v)
			
			return p

		if self.method == 'averaging':
			return np.sum(vectors)/len(vectors)
