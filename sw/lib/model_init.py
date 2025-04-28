import torch.nn as nn
from torch.nn import init

__ALL__ = [
    'init_params',
]


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
