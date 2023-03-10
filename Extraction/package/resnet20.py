import torch
import torch.nn as nn

## ResNet 
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# add batch norm
class ResNet20_basic(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.conv3 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.conv4 = BasicConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        
        self.fc = nn.Linear(512*14*14 , num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
        layers = []
        layers.append(block(planes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)           # 112x112
        x = self.layer1(x)          # 
        x = self.conv2(x)           # 56x56
        x = self.layer2(x)          # 
        x = self.conv3(x)           # 28x28
        x = self.layer3(x)          # 
        x = self.conv4(x)           # 14x14
        x = self.layer4(x)          # 

        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        x = self.fc(x)

        return x
    
def resnet20_basic(num_classes):
    layers=[1, 2, 4, 1]
    model = ResNet20_basic(BasicBlock, layers, num_classes)
    return model