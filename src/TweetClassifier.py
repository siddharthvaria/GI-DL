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

class TweetClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers = 2, dropout = 0.5, lr = 0.01):
        super(TweetClassifier, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        print 'output size: ', self.output_size
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
#         return (autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size)),
#                 autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size)))


def get_training_example(x_i, y_i, evaluation = False):
    x_i = autograd.Variable(torch.LongTensor(x_i).cuda(), volatile = evaluation)
    y_i = autograd.Variable(torch.LongTensor([y_i]).cuda())
#     x_i = autograd.Variable(torch.LongTensor(x_i), volatile = evaluation)
#     y_i = autograd.Variable(torch.LongTensor([y_i]))
    return x_i, y_i

def time_since(since):
    s = time.time() - since
    m = math.floor(float(s) / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def save(clf, model_save_dir):
    save_filename = os.path.join(model_save_dir, 'model.pt')
    torch.save(clf, save_filename)
    print('Saved as %s' % save_filename)

def fit(clf, X, y):

    clf.train()

    train_loss = 0

    for x_i, y_i in zip(X, y):

        x_i, y_i = get_training_example(x_i, y_i)

        hidden = clf.init_hidden()

        clf.zero_grad()

        for i in range(len(x_i)):
            output, hidden = clf(x_i[i], hidden)

#         print 'len(output): ', len(output)
#         print 'len(output[0]): ', len(output[0])

        loss = clf.criterion(output, y_i)
        loss.backward()
        clf.decoder_optimizer.step()
        train_loss += loss.data[0]

    train_loss /= len(X)
    return train_loss

def predict(clf, X, y):

    clf.eval()

    test_loss = 0

    correct = 0

    for x_i, y_i in zip(X, y):

        x_i, y_i = get_training_example(x_i, y_i, evaluation = True)

        hidden = clf.init_hidden()

        for i in range(len(x_i)):
            output, hidden = clf(x_i[i], hidden)

        loss = clf.criterion(output, y_i)

        test_loss += loss.data[0]

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(y_i.data).cpu().sum()

    test_loss /= len(X)

    return test_loss, correct

def main(train_file, val_file, test_file, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate):

    corpus = TweetCorpus(train_file, val_file, test_file)

    print 'len(vocab): ', len(corpus.char2idx)
    print 'len(label): ', len(corpus.label2idx)

    X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_stratified_splits(split_ratio = 0.1)

    clf = TweetClassifier(len(corpus.char2idx), hidden_size, len(corpus.label2idx), n_layers = n_layers, dropout = dropout, lr = learning_rate)
    clf = clf.cuda()

    # start = time.time()
    # all_losses = []


#     if epoch % print_every == 0:
#         print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, float(epoch) / n_epochs * 100, loss))
#
#     if epoch % plot_every == 0:
#         all_losses.append(loss_avg / plot_every)
#         loss_avg = 0

    # Loop over epochs.
    # best_val_loss = None
    # best_test_loss = None
    best_val_acc = None
    best_test_acc = None

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        fit(clf, X_train, y_train)
        val_loss, val_acc = predict(clf, X_val, y_val)
        print('-' * 89)
        print('| End of epoch {:3d} | training time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format
              (val_loss, val_acc, len(X_val), 100. * val_acc / len(X_val)))
        # Save the model if the validation accuracy is the best we've seen so far.
        if not best_val_acc or val_acc > best_val_acc:
            best_val_acc = val_acc
            test_loss, test_acc = predict(clf, X_test, y_test)
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format
                  (test_loss, test_acc, len(X_test), 100. * test_acc / len(X_test)))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save(clf, model_save_dir)
        print('-' * 89)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('train_file', type = str)
    parser.add_argument('val_file', type = str)
    parser.add_argument('test_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('--n_epochs', type = int, default = 50)
    parser.add_argument('--hidden_size', type = int, default = 128)
    parser.add_argument('--n_layers', type = int, default = 2)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 0.01)

    args = vars(parser.parse_args())

    main(**args)
