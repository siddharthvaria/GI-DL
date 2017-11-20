import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import time
import math
import os
import cPickle
import numpy as np
from data_utils.TweetReader1 import TweetCorpus
from Datasets import Dataset1
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)

class TweetClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers = 2, dropout = 0.5, lr = 0.01):
        super(TweetClassifier, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size, padding_idx = 0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout = dropout)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.decoder_optimizer = optim.Adam(self.parameters(), lr = lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inp, hidden):
        # input (seq_len, batch, input_size): tensor containing the features of the input sequence.
#         print 'inp.size(): ', inp.size()
        emb = self.encoder(inp.t())
#         print 'emb.size(): ', emb.size()
        emb = self.drop(emb)
	# output (seq_len, batch, hidden_size * num_directions): tensor containing the
        # output features (h_t) from the last layer of the RNN, for each t.
        # output, hidden = self.lstm(emb.view(self.max_seq_len, self.batch_size, -1), hidden)
        output, hidden = self.lstm(emb, hidden)
#        print 'output1.size(): ', output.size()
        output = self.drop(output[-1])
#        print 'output2.size(): ', output.size()
        output = self.decoder(output)
#        print 'output3.size(): ', output.size()
        return output, hidden

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda()),
                autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda()))
#         return (autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size)),
#                 autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size)))

def get_variables(data, target, evaluation = False):
#     print 'type(data):', type(data)
#     print 'type(data[0]):', type(data[0])
#     print 'type(target): ', type(target)
#     print 'len(data):', len(data)
#     print 'len(data[0]):', len(data[0])
    data = autograd.Variable(data.cuda(), volatile = evaluation)
    target = autograd.Variable(target.cuda())
    return data, target

def time_since(since):
    s = time.time() - since
    m = math.floor(float(s) / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def save(clf, model_save_dir):
    save_filename = os.path.join(model_save_dir, 'classifier_model.pt')
    torch.save(clf, save_filename)
    print('Saved as %s' % save_filename)

def fit(clf, data_loader):

    train_loss = 0

    count = 0

    for (data, target) in data_loader:

        # data has dimension: batch_size, seq_len
        # target had dimension: batch_size
        clf.train()

        clf.decoder_optimizer.zero_grad()

        data, target = get_variables(data, target)

        batch_size = data.size()[0]

        hidden = clf.init_hidden(batch_size)

        output, hidden = clf(data, hidden)

        # At this point, the output has dimension: batch_size, number_classes
        # At this point, target has dimension: batch_size
        loss = clf.criterion(output, target)
        loss.backward()
        clf.decoder_optimizer.step()
        train_loss += (loss.data[0] * batch_size)
        count += batch_size

    train_loss /= count
    return train_loss

def predict(clf, data_loader):

    predictions = []
    targets = []

    test_loss = 0

    count = 0

    correct = 0

    for data, target in data_loader:

        clf.eval()

        data, target = get_variables(data, target, evaluation = True)

        batch_size = data.size()[0]

        hidden = clf.init_hidden(batch_size)

        output, hidden = clf(data, hidden)

        loss = clf.criterion(output, target)

        test_loss += (loss.data[0] * batch_size)

        # output is of type: <class 'torch.autograd.variable.Variable'>
        # output.data is of type: <class 'torch.cuda.FloatTensor'>
        # output.data.max(1) gives both the x and y co-ordinates so output.data.max(1)[1] gives the y co-ordinates.
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        pred = pred.view(batch_size)

        correct += pred.eq(target.data).cpu().sum()

        predictions += pred.cpu().numpy().tolist()
        targets += target.data.cpu().numpy().tolist()

        count += batch_size

    test_loss /= count
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    return test_loss, (100.0 * float(correct)) / len(predictions), predictions, targets

def plot_confusion_matrix(cm, class_names, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation = 45)
    plt.yticks(tick_marks, class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_losses(model_save_dir, train_losses, val_losses):

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.savefig(os.path.join(model_save_dir, 'train_loss.png'))
    plt.figure()
    plt.plot(range(1, len(val_losses) + 1), val_losses)
    plt.savefig(os.path.join(model_save_dir, 'val_loss.png'))

def print_class_distributions(class_names, train_label_dist, val_label_dist, test_label_dist):

    print 'Train set distribution:'
    for k, v in train_label_dist.iteritems():
        print class_names[k], v
    print '#####################################'
    print 'Val set distribution:'
    for k, v in val_label_dist.iteritems():
        print class_names[k], v
    print '#####################################'
    print 'Test set distribution:'
    for k, v in test_label_dist.iteritems():
        print class_names[k], v
    print '#####################################'

def main(train_file, val_file, test_file, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate, batch_size):

    corpus = TweetCorpus(train_file, val_file, test_file)

    class_names = corpus.get_class_names()

    print 'len(vocab): ', len(corpus.char2idx)
    print 'len(label): ', len(corpus.label2idx)

    # X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_stratified_splits()
    X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_splits()

    train_label_dist, val_label_dist, test_label_dist = corpus.get_label_dist(y_train, y_val, y_test)

    print_class_distributions(class_names, train_label_dist, val_label_dist, test_label_dist)

    train_dataset = Dataset1(X_train, y_train)
    val_dataset = Dataset1(X_val, y_val)
    test_dataset = Dataset1(X_test, y_test)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, **kwargs)

    clf = TweetClassifier(len(corpus.char2idx) + 1, hidden_size, len(corpus.label2idx), n_layers = n_layers, dropout = dropout, lr = learning_rate)
    clf = clf.cuda()

    best_val_acc = 0
    best_test_acc = 0
    best_test_predictions = None
    best_test_targets = None

    train_losses = []
    val_losses = []

    patience = 5

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loss = fit(clf, train_loader)
        train_losses.append(train_loss)
        print('| End of epoch {:3d} | training time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        _, train_acc, _, _ = predict(clf, train_loader)
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format
              (train_loss, train_acc, len(X_train), 100. * train_acc / len(X_train)))
        val_loss, val_acc, _, _ = predict(clf, val_loader)
        val_losses.append(val_loss)
        print('Val set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(val_loss, val_acc))
        # Save the model if the validation accuracy is the best we've seen so far.
        if val_acc > best_val_acc:
            patience = 5
            best_val_acc = val_acc
            test_loss, test_acc, predictions, targets = predict(clf, test_loader)
            print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(test_loss, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_predictions = predictions
                best_test_targets = targets
                save(clf, model_save_dir)
        else:
            patience -= 1

        if patience <= 0:
            print 'Early stopping . . .'
            break

    np.set_printoptions(precision = 2)
    cnf_matrix = confusion_matrix(best_test_targets, best_test_predictions)
    # cPickle.dump(cnf_matrix, open('confusion_matrix.p', 'wb'))
    print(classification_report(best_test_targets, best_test_predictions, target_names = class_names))

    plot_confusion_matrix(cnf_matrix, class_names, normalize = False, title = 'Confusion matrix')
    # plt.show()

    plt.savefig(os.path.join(model_save_dir, 'confusion_matrix_best_model.png'))

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.savefig(os.path.join(model_save_dir, 'train_loss.png'))
    plt.figure()
    plt.plot(range(1, len(val_losses) + 1), val_losses)
    plt.savefig(os.path.join(model_save_dir, 'val_loss.png'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('train_file', type = str)
    parser.add_argument('val_file', type = str)
    parser.add_argument('test_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('--n_epochs', type = int, default = 15)
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--n_layers', type = int, default = 2)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--batch_size', type = int, default = 128)

    args = vars(parser.parse_args())

    main(**args)
