import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import time
import math
import os
import numpy as np
from TweetReader import TweetCorpus
from Datasets import Dataset1, Dataset2
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)

class HybridModel(nn.Module):

    def __init__(self, mode, input_size, hidden_size, nlabels1, nlabels2, n_layers = 2, dropout = 0.5, lr = 0.01):

        super(HybridModel, self).__init__()
        self.mode = mode
        self.drop = nn.Dropout(dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlabels1 = nlabels1
        self.nlabels2 = nlabels2
        # print 'number of classes: ', self.output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(self.input_size, self.hidden_size, padding_idx = 0)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, dropout = dropout)
        # decoder1 is for lm and decoder2 is for classifier
        self.decoder1 = nn.Linear(self.hidden_size, self.nlabels1)
        self.decoder2 = nn.Linear(self.hidden_size, self.nlabels2)

    def forward(self, inp, hidden):
        emb = self.encoder(inp.t())
        emb = self.drop(emb)
        output, hidden = self.lstm(emb, hidden)
        # At this point, the output has dimension: seq_len, batch_size, hidden_size
        # We reshape it to have dimension: batch_size * seq_len, hidden_size
        if self.mode == 'lm':
            output = output.transpose(0, 1)
            output = torch.unbind(output, 0)
            output = torch.cat(output, 0)
            # print 'reshaped output size: ', output.size()
            output = self.drop(output)
            output = self.decoder1(output)
        elif self.mode == 'clf':
            output = self.drop(output[-1])
            output = self.decoder2(output)

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

def save(model, model_save_name, model_save_dir, save_type = 1):
    save_filename = os.path.join(model_save_dir, model_save_name + '.pt')
    if save_type == 1:
        torch.save(model, save_filename)
    else:
        torch.save(model.state_dict(), save_filename)
    print('Saved as %s' % save_filename)

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

def train_lm(corpus, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate, batch_size):

    def fit(lm, opt, criterion, data_loader):

        train_loss = 0

        count = 0

        batches_processed = 0

        for (data, target) in data_loader:

            lm.train()

            batches_processed += 1
            # data has dimension: batch_size, seq_len
            # target had dimension: batch_size

            opt.zero_grad()

            data, target = get_variables(data, target)

            bsz = data.size()[0]

            hidden = lm.init_hidden(bsz)

            output, hidden = lm(data, hidden)

            # At this point, the output has dimension: batch_size, number_classes
            # At this point, target has dimension: batch_size
            target_size = target.size()
            # print 'target_size: ', target_size
            loss = criterion(output, target.view(target_size[0] * target_size[1]))
            loss.backward()
            opt.step()
            train_loss += (loss.data[0] * target_size[0] * target_size[1])
            count += (target_size[0] * target_size[1])

            if batches_processed % (total_batches / 4) == 0:
                print 'Batches processed so far: ', batches_processed

        train_loss /= count
        return train_loss

    save_type = 2

    model1 = HybridModel('lm', len(corpus.char2idx) + 1, hidden_size, len(corpus.char2idx) + 1, len(corpus.label2idx), n_layers = n_layers, dropout = dropout, lr = learning_rate)
    model1 = model1.cuda()

    optimizer = optim.Adam(model1.parameters(), lr = learning_rate)

    loss_criterion = nn.CrossEntropyLoss()

    X_train = corpus.get_splits_for_lm()

    total_batches = len(X_train) / batch_size

    print 'Total number of batches: ', total_batches

    train_dataset = Dataset2(X_train)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, **kwargs)

    best_train_loss = 1000000
    patience = 3

    train_losses = []

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loss = fit(model1, optimizer, loss_criterion, train_loader)
        train_losses.append(train_loss)
        print('| End of epoch {:3d} | training time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
#         _, train_acc, _, _ = predict(model, train_loader)
        print('Train set: Average loss: {:.4f}'.format(train_loss))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience = 3
            # Save the model if the validation accuracy is the best we've seen so far.
            save(model1, 'language_model', model_save_dir, save_type = save_type)
        else:
            patience -= 1

        if patience <= 0:
            print 'Early stopping . . .'
            break

def train_classifier(corpus, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate, batch_size):

    def fit(clf, opt, criterion, data_loader):

        train_loss = 0

        count = 0

        for (data, target) in data_loader:

            # data has dimension: batch_size, seq_len
            # target had dimension: batch_size
            clf.train()

            opt.zero_grad()

            data, target = get_variables(data, target)

            bsz = data.size()[0]

            hidden = clf.init_hidden(bsz)

            output, hidden = clf(data, hidden)

            # At this point, the output has dimension: batch_size, number_classes
            # At this point, target has dimension: batch_size
            loss = criterion(output, target)
            loss.backward()
            opt.step()
            train_loss += (loss.data[0] * bsz)
            count += bsz

        train_loss /= count
        return train_loss

    def predict(clf, criterion, data_loader):

        predictions = []
        targets = []

        test_loss = 0

        count = 0

        correct = 0

        for data, target in data_loader:

            clf.eval()

            data, target = get_variables(data, target, evaluation = True)

            bsz = data.size()[0]

            hidden = clf.init_hidden(bsz)

            output, hidden = clf(data, hidden)

            loss = criterion(output, target)

            test_loss += (loss.data[0] * bsz)

            # output is of type: <class 'torch.autograd.variable.Variable'>
            # output.data is of type: <class 'torch.cuda.FloatTensor'>
            # output.data.max(1) gives both the x and y co-ordinates so output.data.max(1)[1] gives the y co-ordinates.
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            pred = pred.view(bsz)

            correct += pred.eq(target.data).cpu().sum()

            predictions += pred.cpu().numpy().tolist()

            targets += target.data.cpu().numpy().tolist()

            count += bsz

        test_loss /= count
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        return test_loss, (100.0 * float(correct)) / len(predictions), predictions, targets

    save_type = 2

    if save_type == 1:
        model2 = torch.load(os.path.join(model_save_dir, 'language_model.pt'))
        # change the mode from lm to clf
        model2.mode = 'clf'
    else:
        model2 = HybridModel('clf', len(corpus.char2idx) + 1, hidden_size, len(corpus.char2idx) + 1, len(corpus.label2idx), n_layers = n_layers, dropout = dropout, lr = learning_rate)
        model2.load_state_dict(torch.load(os.path.join(model_save_dir, 'language_model.pt')))

    model2 = model2.cuda()

    optimizer = optim.Adam(model2.parameters(), lr = learning_rate)

    loss_criterion = nn.CrossEntropyLoss()

    class_names = corpus.get_class_names()

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

    best_val_acc = 0
    best_test_acc = 0
    best_test_predictions = None
    best_test_targets = None

    train_losses = []
    val_losses = []

    patience = 5

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loss = fit(model2, optimizer, loss_criterion, train_loader)
        train_losses.append(train_loss)
        print('| End of epoch {:3d} | training time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        _, train_acc, _, _ = predict(model2, loss_criterion, train_loader)
        print('Train set: Average loss: {:.4f}, Accuracy:{:.0f}%'.format
              (train_loss, train_acc))
        val_loss, val_acc, _, _ = predict(model2, loss_criterion, val_loader)
        val_losses.append(val_loss)
        print('Val set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format
              (val_loss, val_acc))
        # Save the model if the validation accuracy is the best we've seen so far.
        if val_acc > best_val_acc:
            patience = 5
            best_val_acc = val_acc
            test_loss, test_acc, predictions, targets = predict(model2, loss_criterion, test_loader)
            print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format
                  (test_loss, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_predictions = predictions
                best_test_targets = targets
                save(model2, 'classifier_model', model_save_dir, save_type = save_type)
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

def main(train_file, val_file, test_file, tweets_file, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate, batch_size):

    corpus = TweetCorpus(train_file, val_file, test_file, tweets_file)

    print 'len(vocab): ', len(corpus.char2idx)

    train_lm(corpus, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate, batch_size)

    # train_classifier(corpus, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate, batch_size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('train_file', type = str)
    parser.add_argument('val_file', type = str)
    parser.add_argument('test_file', type = str)
    parser.add_argument('tweets_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('--n_epochs', type = int, default = 15)
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--n_layers', type = int, default = 2)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--batch_size', type = int, default = 128)

    args = vars(parser.parse_args())

    main(**args)
