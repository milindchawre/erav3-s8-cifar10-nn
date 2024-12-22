import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    """
    Receptive Field calculation for current architecture:
    Layer               RF      n_in    j_in    n_out   j_out   k   d   s   p
    Input               1       32      1       32      1       -   -   -   -
    
    Conv1.1             3       32      1       32      1       3   1   1   1
    Conv1.2             5       32      1       32      1       3   1   1   1
    Conv1.3(s2)         9       32      1       16      2       3   1   2   1
    
    DWConv2.1           13      16      2       16      2       3   1   1   1
    DWConv2.2           17      16      2       16      2       3   1   1   1
    DWConv2.3(s2)       25      16      2       8       4       3   1   2   1
    
    Conv3.1(d2)         41      8       4       8       4       3   2   1   2
    Conv3.2(d4)         73      8       4       8       4       3   4   1   4
    Conv3.3(s2)         89      8       4       4       8       3   1   2   1
    
    Conv4.1(g4)         105     4       8       4       8       3   1   1   1
    Conv4.2(g4)         121     4       8       4       8       3   1   1   1
    Conv4.3(s2)         153     4       8       2       16      3   1   2   1

    Final RF: 153x153 (>44 requirement)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self._create_conv1()
        self.conv2 = self._create_conv2()
        self.conv3 = self._create_conv3()
        self.conv4 = self._create_conv4()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(88, num_classes)

    def _create_conv1(self):
        return nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

    def _create_conv2(self):
        return nn.Sequential(
            DepthwiseSeparableConv(48, 48, 3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.05),
            DepthwiseSeparableConv(48, 48, 3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.05),
            DepthwiseSeparableConv(48, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

    def _create_conv3(self):
        return nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, groups=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4, groups=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv2d(64, 72, 3, stride=2, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

    def _create_conv4(self):
        return nn.Sequential(
            nn.Conv2d(72, 72, 3, padding=1, groups=4),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv2d(72, 72, 3, padding=1, groups=4),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv2d(72, 88, 3, stride=2, padding=1),
            nn.BatchNorm2d(88),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        identity = x
        x = self.conv3[0:4](x)  # First conv block with dropout
        x = x + identity
        identity = x
        x = self.conv3[4:8](x)  # Second conv block with dropout
        x = x + identity
        x = self.conv3[8:](x)   # Third conv block with dropout
        
        identity = x
        x = self.conv4[0:4](x)  # First conv block with dropout
        x = x + identity
        identity = x
        x = self.conv4[4:8](x)  # Second conv block with dropout
        x = x + identity
        x = self.conv4[8:](x)   # Third conv block with dropout
        
        x = self.gap(x)
        x = self.dropout(x)
        x = x.view(-1, 88)
        x = self.fc(x)
        return x 