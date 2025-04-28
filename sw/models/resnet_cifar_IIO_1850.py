
import torch
import torch.nn as nn
from quant import a_bits, w_bits
from lib.model_init import init_params

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
]


# model_urls = {
#     'resnet18':
#         '{}/resnetforcifar/resnet18-cifar-acc78.41.pth'.format(
#             pretrained_models_path),
#     'resnet34':
#         '{}/resnetforcifar/resnet34-cifar-acc78.84.pth'.format(
#             pretrained_models_path),
#     'resnet50':
#         '{}/resnetforcifar/resnet50-cifar-acc77.88.pth'.format(
#             pretrained_models_path),
#     'resnet101':
#         '{}/resnetforcifar/resnet101-cifar-acc80.16.pth'.format(
#             pretrained_models_path),
#     'resnet152':
#         '{}/resnetforcifar/resnet152-cifar-acc80.99.pth'.format(
#             pretrained_models_path),
# }


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.act = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        # ReLU

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        # ReLU

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        # ReLU

    def forward(self, x):
        residual = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, norm_layer=None, inter_layer=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inter_layer = inter_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)  # Attention! Here bias is True

    def _make_layer(self, block, planes, num_blocks, stride=1):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            planes: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
            # self.inplanes = planes * block.expansion  # not used for block.expansion==1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.inter_layer:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x = self.avg_pool(x4)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return [x1, x2, x3, x4, x]
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    def inference(self, x):
        x = x * ((2 ** a_bits - 1) * (2 ** w_bits - 1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x / ((2 ** a_bits - 1) * (2 ** w_bits - 1))
        return x


def _resnet(block, layers, pretrained, **kwargs):
    arch = kwargs.pop('arch')
    model = ResNet(block, layers, **kwargs)
    # only load state_dict()
    # if pretrained:
    #     model.load_state_dict(
    #         torch.load(model_urls[arch], map_location=torch.device('cpu')))
    return model


def resnet18(pretrained=False, **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], pretrained, arch='resnet18', **kwargs)


def resnet34(pretrained=False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], pretrained, arch='resnet34', **kwargs)


def resnet50(pretrained=False, **kwargs):
    return _resnet(BottleNeck, [3, 4, 6, 3], pretrained, arch='resnet50', **kwargs)


def resnet101(pretrained=False, **kwargs):
    return _resnet(BottleNeck, [3, 4, 23, 3], pretrained, arch='resnet101', **kwargs)


def resnet152(pretrained=False, **kwargs):
    return _resnet(BottleNeck, [3, 8, 36, 3], pretrained, arch='resnet152', **kwargs)


if __name__ == '__main__':
    net = resnet18(num_classes=10)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = net(dummy_input)
    print(output)
    print(output.size())

