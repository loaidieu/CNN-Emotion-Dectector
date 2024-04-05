from utils.tools_lib import *

torch.manual_seed(0) # reproducibility

class VarCnn1(torch.nn.Module):
    def __init__(self):
        super(VarCnn1, self).__init__()

        # first convolutional layer with 3x3 filter size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=3//2)
        self.bn1   = torch.nn.BatchNorm2d(16)                                                         # batch normalization layer
        self.relu1 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # second convolutional layer with 5x5 filter size
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=5//2)
        self.bn2   = torch.nn.BatchNorm2d(32)                                                         # batch normalization layer
        self.relu2 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layer
        self.flatten = torch.nn.Flatten()
        self.linear  = torch.nn.Linear(in_features=32*12*12, out_features=4)                            # since input image size is 48x48

        # dropout rate
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x