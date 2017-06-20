from torch.utils.data import Dataset

class Dataset1(Dataset):

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

class Dataset2(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (X[index][:-1],X[index][1:]) where X is the list of sequences
        """
        return (self.X[index][:-1], self.X[index][1:])
