import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import time
import math
import os
import numpy as np
from random import randint
from TweetReader import TweetCorpus
from utils import Dataset2
from torch.utils.data import DataLoader

torch.manual_seed(1)
torch.cuda.manual_seed(1)

class TweetLanguageModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers = 2, dropout = 0.5, lr = 0.01):

        super(TweetLanguageModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # print 'number of classes: ', self.output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size, padding_idx = 0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout = dropout)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.decoder_optimizer = optim.Adam(self.parameters(), lr = lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inp, hidden):
        emb = self.encoder(inp.t())
        emb = self.drop(emb)
        output, hidden = self.lstm(emb, hidden)
        # At this point, the output has dimension: seq_len, batch_size, hidden_size
        # We reshape it to have dimension: batch_size * seq_len, hidden_size
        output = output.transpose(0, 1)
        output = torch.unbind(output, 0)
        output = torch.cat(output, 0)
        # print 'reshaped output size: ', output.size()
        output = self.drop(output)
        output = self.decoder(output)
        # print 'final output size: ', output.size()
        return output, hidden

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda()),
                autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda()))


def get_variables(data, target, evaluation = False):
    data = autograd.Variable(data.cuda(), volatile = evaluation)
    target = autograd.Variable(target.cuda())
    return data, target

def time_since(since):
    s = time.time() - since
    m = math.floor(float(s) / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def save(lm, model_save_dir):
    save_filename = os.path.join(model_save_dir, 'language_model.pt')
    torch.save(lm, save_filename)
    print('Saved as %s' % save_filename)

def fit(lm, data_loader, total_batches):

    train_loss = 0

    count = 0

    batches_processed = 0

    for (data, target) in data_loader:

        batches_processed += 1
        # data has dimension: batch_size, seq_len
        # target had dimension: batch_size
        lm.train()

        lm.decoder_optimizer.zero_grad()

        data, target = get_variables(data, target)

        batch_size = data.size()[0]

        hidden = lm.init_hidden(batch_size)

        output, hidden = lm(data, hidden)

        target_size = target.size()
        # print 'target_size: ', target_size
        loss = lm.criterion(output, target.view(target_size[0] * target_size[1]))
        loss.backward()
        lm.decoder_optimizer.step()
        train_loss += (loss.data[0] * target_size[0] * target_size[1])
        count += (target_size[0] * target_size[1])

        if batches_processed % (total_batches / 4) == 0:
            print 'Batches processed so far: ', batches_processed

    train_loss /= count
    return train_loss

def generate(lm, corpus, start_letter = 'A'):
    start_letter_index = corpus.char2idx[start_letter]
    inp = autograd.Variable(torch.LongTensor(np.asarray([[start_letter_index]])).cuda(), volatile = True)
    hidden = lm.init_hidden(1)
    output_name = start_letter
    max_length = randint(50, 140)
    for _ in range(max_length):
        lm.eval()
        output, hidden = lm(inp, hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        letter = corpus.idx2char[topi]
        output_name += letter
        inp = autograd.Variable(torch.LongTensor(np.asarray([[topi]])).cuda(), volatile = True)

    return output_name

def main(tweets_file, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate, batch_size):

    corpus = TweetCorpus(unlabeled_tweets_file = tweets_file)

    print 'len(vocab): ', len(corpus.char2idx)

    X_train = corpus.get_splits_for_lm()

    print 'Total number of batches: ', len(X_train) / batch_size

    train_dataset = Dataset2(X_train)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, **kwargs)

    # input_size, hidden_size, output_size, n_layers = 2, dropout = 0.5, lr = 0.01
    lm = TweetLanguageModel(len(corpus.char2idx) + 1, hidden_size, len(corpus.char2idx) + 1, n_layers = n_layers, dropout = dropout, lr = learning_rate)
    lm = lm.cuda()

    train_losses = []

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loss = fit(lm, train_loader, len(X_train) / batch_size)
        train_losses.append(train_loss)
        print('| End of epoch {:3d} | training time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
#         _, train_acc, _, _ = predict(lm, train_loader)
        print('Train set: Average loss: {:.4f}'.format(train_loss))
        # Save the model if the validation accuracy is the best we've seen so far.
        save(lm, model_save_dir)

    lm_loaded = torch.load(os.path.join(model_save_dir, 'language_model.pt'))

    start_letters = ['@', 'r']
    for i in range(len(start_letters)):
        op = generate(lm_loaded, corpus, start_letter = start_letters[i])
        print op

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('tweets_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('--n_epochs', type = int, default = 20)
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--n_layers', type = int, default = 2)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--batch_size', type = int, default = 128)

    args = vars(parser.parse_args())

    main(**args)
