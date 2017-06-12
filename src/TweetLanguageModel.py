import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import time
import math
import os
from TweetReader import TweetCorpus

torch.manual_seed(1)
torch.cuda.manual_seed(1)

class TweetLanguageModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers = 2, dropout = 0.5, lr = 0.01):
        super(TweetLanguageModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout = dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.decoder_optimizer = optim.Adam(self.parameters(), lr = lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inp, hidden):
        emb = self.encoder(inp.view(1, -1))
        emb = self.drop(emb)
        # inp = self.encoder(inp.view(1, -1))
        output, hidden = self.lstm(emb.view(1, 1, -1), hidden)
        output = self.drop(output)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size).cuda()),
                autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size).cuda()))

def get_training_example(seq):
    inp = autograd.Variable(torch.LongTensor(seq[:-1]).cuda())
    target = autograd.Variable(torch.LongTensor(seq[1:]).cuda())
    return inp, target

def fit(decoder, inp, target):

    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(len(inp)):
        output, hidden = decoder(inp[c], hidden)
        loss += decoder.criterion(output, target[c])

    loss.backward()
    decoder.decoder_optimizer.step()

    return loss.data[0] / len(inp)

def time_since(since):
    s = time.time() - since
    m = math.floor(float(s) / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def save(decoder, model_save_dir):
    save_filename = os.path.join(model_save_dir, 'model.pt')
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def train(corpus, model_save_dir, n_epochs = 50, hidden_size = 100, n_layers = 2, dropout = 0.5, lr = 0.01):

    print_every = 1
    plot_every = 10

    decoder = TweetLanguageModel(len(corpus.char2idx), hidden_size, len(corpus.char2idx), n_layers = n_layers, dropout = dropout, lr = lr)
    decoder = decoder.cuda()

    start = time.time()
    all_losses = []
    loss_avg = 0

    for epoch in range(1, n_epochs + 1):

        for tweet in corpus.tweets:
            inp, target = get_training_example(corpus.tweet2Indices(tweet))
            loss = fit(decoder, inp, target)
            loss_avg += loss

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, float(epoch) / n_epochs * 100, loss))

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0

    print("Saving...")
    save(decoder, model_save_dir)

def main(tweets_file, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate):

    corpus = TweetCorpus(tweets_file)
    print 'vocab: ', len(corpus.char2idx)
    train(corpus, model_save_dir, n_epochs = n_epochs, hidden_size = hidden_size, n_layers = n_layers, dropout = dropout, lr = learning_rate)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('tweets_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('--n_epochs', type = int, default = 50)
    parser.add_argument('--hidden_size', type = int, default = 100)
    parser.add_argument('--n_layers', type = int, default = 2)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 0.01)

    args = vars(parser.parse_args())

    main(**args)
