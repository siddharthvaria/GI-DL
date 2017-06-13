import csv
from torch.utils.data import Dataset
import torch

def unicode_csv_reader1(utf8_data, **kwargs):
    csv_reader = csv.reader(utf8_data, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def unicode_csv_reader2(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {k:unicode(v, 'utf-8') for k, v in row.iteritems()}

def test_unicode_csv_reader():
    filename = '../data/csv_utf8_test.csv'
    reader = unicode_csv_reader2(open(filename))
    for line in reader:
        print len(line)
        print line

class Dataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (tweet, target) where target is index of the target class.
        """
        return (self.X[index], self.y[index])
