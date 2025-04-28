import torch
import torch.nn as nn
from torch.nn import init


class MyAlexNet(nn.Module):

    # def __init__(self, classes=100):
    #     super(MyAlexNet, self).__init__()
    #     self.prepare = nn.Sequential(
    #         nn.Conv2d(3, 96, kernel_size=3, stride=1, bias=False),  # 3*32*32-->96*30*30
    #         nn.ReLU(inplace=True),
    #     )
    #     self.features_extract = nn.Sequential(
    #         nn.MaxPool2d(kernel_size=3, stride=2),  # -->96*14*14
    #
    #         nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False),  # -->256*9*9
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),  # -->256*4*4
    #
    #         nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False),  # -->256*3*3
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),  # -->384*3*3
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),  # -->256*3*3
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2)  # -->256*2*2=1024
    #     )
    #     self.classifier = nn.Sequential(
    #         # nn.Dropout(),
    #         nn.Linear(1024, 2048, bias=False),
    #         nn.ReLU(inplace=True),
    #         # nn.Dropout(),
    #         nn.Linear(2048, 1024, bias=False),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(1024, classes, bias=False)
    #     )
    #
    # def forward(self, x):
    #     x = self.prepare(x)
    #     x = self.features_extract(x)
    #     # x = torch.flatten(x, 1)
    #     # x = torch.flatten(x, 0)
    #     x = x.view(x.shape[0], -1)
    #     # print(x.shape)
    #     x = self.classifier(x)
    #     return x

    def __init__(self, classes=100):
        super(MyAlexNet, self).__init__()

        self.prepare = nn.Conv2d(3, 96, kernel_size=3, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(96)
        self.prepare_relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.features_extract_1 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.features_extract_1_relu = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.features_extract_2 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(384)
        self.features_extract_2_relu = nn.ReLU(inplace=True)
        self.features_extract_3 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(384)
        self.features_extract_3_relu = nn.ReLU(inplace=True)
        self.features_extract_4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.features_extract_4_relu = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.classifier_1 = nn.Linear(1024, 2048, bias=False)
        self.classifier_1_relu = nn.ReLU(inplace=True)
        self.classifier_2 = nn.Linear(2048, 1024, bias=False)
        self.classifier_2_relu = nn.ReLU(inplace=True)
        self.classifier_3 = nn.Linear(1024, classes, bias=False)

    def forward(self, x):
        x = self.prepare(x)
        x = self.bn0(x)
        x = self.prepare_relu(x)

        x = self.maxpool1(x)
        x = self.features_extract_1(x)
        x = self.bn1(x)
        x = self.features_extract_1_relu(x)

        x = self.maxpool2(x)
        x = self.features_extract_2(x)
        x = self.bn2(x)
        x = self.features_extract_2_relu(x)
        x = self.features_extract_3(x)
        x = self.bn3(x)
        x = self.features_extract_3_relu(x)
        x = self.features_extract_4(x)
        x = self.bn4(x)
        x = self.features_extract_4_relu(x)

        x = self.maxpool3(x)

        x = x.view(x.shape[0], -1)

        x = self.classifier_1(x)
        x = self.classifier_1_relu(x)
        x = self.classifier_2(x)
        x = self.classifier_2_relu(x)
        x = self.classifier_3(x)

        return x

    def inference(self, x):
        x = self.prepare(x)
        x = x * ((2 ** 8 - 1) ** 2)
        x = self.features_extract(x)
        # x = torch.flatten(x, 1)
        # x = torch.flatten(x, 0)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.classifier(x)
        x = x / ((2 ** 8 - 1) ** 2)
        return x


def init_params(net):
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')  # for quanted model using this
            # init.normal_(module.weight, 0, 0.01)  # for float model using this
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm)):
            init.normal_(module.weight, 1.0, 1E-2)
            init.uniform_(module.bias, -1E-2, 1E-2)


def alexnet():
    net = MyAlexNet()
    init_params(net)
    return net


if __name__ == '__main__':
    net = MyAlexNet()
    print(net)
    x = torch.randn(10, 3, 32, 32)
    y = net(x)
    print(y.shape)
