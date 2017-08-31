# from sklearn.manifold import TSNE
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cPickle as pickle
from sklearn import decomposition

def vizualize_embeddings(unicode_embs, unicode_chars):

    model = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress = True)
    trans = model.fit_transform(unicode_embs)

    # plot the emoji using TSNE
    fig = plt.figure()
    ax = fig.add_subplot(111)
#     tsne = man.TSNE(perplexity = 50, n_components = 2, init = 'random', n_iter = 300000, early_exaggeration = 1.0,
#                     n_iter_without_progress = 1000)
#     trans = tsne.fit_transform(V)
    x, y = zip(*trans)
    plt.scatter(x, y, marker = 'o', alpha = 0.0, s = 150)

    for i in range(len(trans)):
        ax.annotate(unicode_chars[i], xy = trans[i], textcoords = 'data')

    plt.grid()
    plt.show()

def load_emoji2vec(e2v_file):
    prefix = 'eoji'
    unicode_chars = []
    emb_matrix = []
    with open(e2v_file, 'r') as fh:
        for line in fh:
            columns = line.split()
            # print len(columns)
            if columns[0].startswith(prefix):
                hex_code = columns[0][len(prefix):]
                if len(hex_code) == 2:
                    hex_code = '00' + hex_code
                # print hex_code
                try:
                    _tmp = '\U000' + hex_code
                    unicode_char = _tmp.decode('unicode-escape')
                except UnicodeDecodeError:
                    _tmp = '\u' + hex_code
                    unicode_char = _tmp.decode('unicode-escape')

                unicode_chars.append(unicode_char)
                _tmp_vec = []
                for v in columns[1:]:
                    _tmp_vec.append(float(v))
                _tmp_vec = np.asarray(_tmp_vec)
                emb_matrix.append(_tmp_vec)

    return np.asarray(emb_matrix), unicode_chars

def dim_reduce(X):
    pca = decomposition.PCA(n_components = 128)
    pca.fit(X)
    X = pca.transform(X)
    return X

def write_emoji_embs(base_path, unicode_embs, unicode_chars):
    dimension = len(unicode_embs[0])
    print 'Dimension of embeddings: ', dimension
    pickle.dump([unicode_chars, unicode_embs], open(os.path.join(base_path, 'emoji_embeddings_' + str(dimension) + '_.p'), "wb"))
    pass

def main():
    fdir, fname = os.path.split(sys.argv[1])
    unicode_embs, unicode_chars = load_emoji2vec(sys.argv[1])
    unicode_embs = dim_reduce(unicode_embs)
    write_emoji_embs(fdir, unicode_embs, unicode_chars)
    vizualize_embeddings(unicode_embs, unicode_chars)

if __name__ == '__main__':
    main()
