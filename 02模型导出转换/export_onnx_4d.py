import torch
import torch.nn as nn

# 1. 原封不动复制你训练代码里的网络结构（必须一模一样）
class SE1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, l = x.shape
        se = self.avg_pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1)
        return x * se

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout1d(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SE1D(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        return self.gelu(out)

class ResNet1D(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            ResidualBlock1D(64, 64), ResidualBlock1D(64, 64), ResidualBlock1D(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2), ResidualBlock1D(128, 128), ResidualBlock1D(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(128, 256, stride=2), ResidualBlock1D(256, 256), ResidualBlock1D(256, 256)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.drop_final = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x).squeeze(-1)
        x = self.drop_final(x)
        return self.fc(x)

# 2. 包装器：把 2D 输出变 4D
class RKNNOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits = self.model(x)              # [1, 11]
        return logits.view(1, 11, 1, 1)     # -> [1, 11, 1, 1]

# 3. 加载模型
model = ResNet1D(num_classes=11)
model.load_state_dict(torch.load("best_rf_resnet_optim3.pt", map_location="cpu"))
model.eval()

wrapper = RKNNOutputWrapper(model).eval()
dummy = torch.randn(1, 4, 128)

# 4. 导出
torch.onnx.export(
    wrapper, dummy, "modulation_resnet_4d.onnx",
    input_names=["features"], output_names=["logits"], opset_version=17
)
print("导出成功: modulation_resnet_4d.onnx (输出 4D)")