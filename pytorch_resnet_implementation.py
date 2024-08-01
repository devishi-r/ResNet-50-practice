import torch
import torch.nn as nn

#Defining 'Residual block', which will be implemented repeatedly throughout the architecture

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        #identity_downsample: represents an additional conv_layer to resize identity mapping dimensions to match those of the generated output
        super(block, self).__init__()
        self.expansion = 4 # Number of channels at the end of every residual block is 4x (notes)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3 , stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample # conv layer to adjust dimensions of identity mapping to match those of generated output
        self.stride = stride

    def forward (self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        #a Adding the identity mapping, after the res block:
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity) #downsample input to add with output of convolutions successfully
        x+=identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes): 
    # block: class defined right above this
    # layers: list telling us how many times 'block' is to be used;
        # eg: ResNet-50: 3, 4, 6, 3, (refer to Table 1 of ResNet paper)
    # image_channels: 3 for RGB, 1 for MNIST, etc.
    # num_classes: data-set dependent; imagenet-mini: 1000+ , CIFAR-10: 10

        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        #ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Define desired output size (1, 1), performs avg pool accordingly
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #Sending through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        # num_residual_blocks: per layer, eg: [3, 4, 6, 3]
        # stride: eg: (3): stride = 1, (4, 6, 3): stride = 2

        # Declaration
        identity_downsample = None
        layers = []

        # Defining identity downsample (have a conv layer to change the identity size/dimensions)
        if stride!=1 or self.in_channels!=out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size = 1, stride = stride),
                                                nn.BatchNorm2d(out_channels*4)) # kernel size = 1 since we're only altering number of channels 
            
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range (num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential (*layers)
    

def ResNet50(img_channel = 3, num_classes = 1000):  
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)

def ResNet101(img_channel = 3, num_classes = 1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)

def ResNet152(img_channel = 3, num_classes = 1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)

def test():
    net = ResNet50() #creates an instance of 'ResNet' class
    x = torch.randn(2, 3, 224, 224) #creates a 4-dimensional tensor with random values
    y = net(x)#.to('cuda') #passes tensor 'x' through network, triggering 'forward' method of 'ResNet' class
    print(y.shape)
    print(y)
    print(net)

test()


#Linear: map features to our desired number of out classes, that helps in final classification      