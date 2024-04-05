from utils.tools_lib import *

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    # method from Dataset class
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])