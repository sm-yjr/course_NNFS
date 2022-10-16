import torch.optim

import customfunc as cf
from torch import nn
from torch.utils.data import DataLoader
from scipy.io import loadmat
from torch.nn import init

# parameter setting
num_epochs = 100
batch_size = 10

# neuron number
num_i, num_h, num_o = 33, 100, 2

# data loading
train_PATH = './data/data_train.mat'
label_PATH = './data/label_train.mat'
test_PATH = './data/data_test.mat'

train_data = loadmat(train_PATH)['data_train']
label_data = loadmat(label_PATH)['label_train']
test_data = loadmat(test_PATH)['data_test']

train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# MLP Modeling
class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLP, self).__init__()
        self.flatten = cf.FlattenLayer()
        self.linear1 = nn.Linear(n_i, n_h)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(n_h, n_o)

    def forward(self, input):
        return self.linear2(self.sigmoid(self.linear1(self.flatten(input))))


net = MLP(num_i, num_h, num_o)
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

# train model
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# cf.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)


