from utils.tools_lib import *

torch.manual_seed(0) # reproducibility

class MainCnn(torch.nn.Module):
    def __init__(
            self,

            # conv layer 1
            cnn_layer1_kernels=16,
            cnn_layer1_kernel_size=3,
            cnn_layer1_padding=None,
            cnn_layer1_poolsize=2,
            cnn_layer1_dropout=0.1,

            # conv layer 2
            cnn_layer2_kernels=32,
            cnn_layer2_kernel_size=5,
            cnn_layer2_padding=None,
            cnn_layer2_poolsize=2,
            cnn_layer2_dropout=0.1,

            # conv layer 3
            cnn_layer3_kernels=64,
            cnn_layer3_kernel_size=7,
            cnn_layer3_padding=None,
            cnn_layer3_poolsize=2,
            cnn_layer3_dropout=0.1,
        ):
        super(MainCnn, self).__init__()

        if cnn_layer1_padding is None: cnn_layer1_padding = cnn_layer1_kernel_size // 2
        if cnn_layer2_padding is None: cnn_layer2_padding = cnn_layer2_kernel_size // 2
        if cnn_layer3_padding is None: cnn_layer3_padding = cnn_layer3_kernel_size // 2

        ##########################################################################################################
        # CONVOLUTIONAL MODULE
        ##########################################################################################################
        self.conv_module = torch.nn.Sequential()

        # 1st convolutional layer
        self.conv_module.add_module(
            'conv1', 
            torch.nn.Conv2d(
                in_channels=1, 
                out_channels=cnn_layer1_kernels,
                kernel_size=cnn_layer1_kernel_size, 
                padding=cnn_layer1_padding)
        )

        self.conv_module.add_module(
            'bn1',
            torch.nn.BatchNorm2d(cnn_layer1_kernels)
        )

        self.conv_module.add_module(
            'relu1',
            torch.nn.ReLU()
        )

        self.conv_module.add_module(
            'pool1',
            torch.nn.MaxPool2d(
                kernel_size=cnn_layer1_poolsize, 
                stride=cnn_layer1_poolsize)
        )

        self.conv_module.add_module(
            'dropout1',
            torch.nn.Dropout(p=cnn_layer1_dropout)
        )

        # 2nd convolutional layer
        self.conv_module.add_module(
            'conv2', 
            torch.nn.Conv2d(
                in_channels=cnn_layer1_kernels, 
                out_channels=cnn_layer2_kernels,
                kernel_size=cnn_layer2_kernel_size, 
                padding=cnn_layer2_padding)
        )

        self.conv_module.add_module(
            'bn2',
            torch.nn.BatchNorm2d(cnn_layer2_kernels)
        )

        self.conv_module.add_module(
            'relu2',
            torch.nn.ReLU()
        )

        self.conv_module.add_module(
            'pool2',
            torch.nn.MaxPool2d(
                kernel_size=cnn_layer2_poolsize, 
                stride=cnn_layer2_poolsize)
        )

        self.conv_module.add_module(
            'dropout2',
            torch.nn.Dropout(p=cnn_layer2_dropout)
        )

        # 3rd convolutional layer
        self.conv_module.add_module(
            'conv3', 
            torch.nn.Conv2d(
                in_channels=cnn_layer2_kernels, 
                out_channels=cnn_layer3_kernels,
                kernel_size=cnn_layer3_kernel_size, 
                padding=cnn_layer3_padding)
        )

        self.conv_module.add_module(
            'bn3',
            torch.nn.BatchNorm2d(cnn_layer3_kernels)
        )

        self.conv_module.add_module(
            'relu3',
            torch.nn.ReLU()
        )

        self.conv_module.add_module(
            'pool3',
            torch.nn.MaxPool2d(
                kernel_size=cnn_layer3_poolsize, 
                stride=cnn_layer3_poolsize)
        )

        self.conv_module.add_module(
            'dropout3',
            torch.nn.Dropout(p=cnn_layer3_dropout)
        )

        ##########################################################################################################
        # DENSE MODULE
        ##########################################################################################################
        # sample input to get the number of features
        sample_input = torch.randn(1, 1, 48, 48)
        sample_output = self.conv_module(sample_input)
        
        self.dense_module = torch.nn.Sequential()

        # flatten
        self.dense_module.add_module(
            'flatten',
            torch.nn.Flatten()
        )

        self.dense_module.add_module(
            'linear',
            torch.nn.Linear(
                in_features=self.num_flat_features(sample_output),
                out_features=4)
        )

        # softmax
        self.dense_module.add_module(
            'softmax',
            torch.nn.LogSoftmax(dim=1)
        )

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s  
        return num_features

    def forward(self, x):
        x = x.float()
        x = self.conv_module(x)
        x = self.dense_module(x)
        return x