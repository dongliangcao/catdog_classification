import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None):
        """
        VGG16

        Args:
            num_classes:
                number of classes to classify, default 2
            pretrained_path:
                path of the model for initialization
        """
        super().__init__()
        self.pretrained_path = pretrained_path
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.features = nn.Sequential(
            self._make_layer(2, 3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_layer(3, 512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # init weights
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layer(self, n_convs, in_channels, out_channels):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            layer += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(len(new_keys) - 2): # except the last linear layer (weight + bias)
                new_dic[new_keys[i]] = dic[keys[i]]
            self.load_state_dict(new_dic)