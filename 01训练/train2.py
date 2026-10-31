import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim.lr_scheduler import LinearLR, SequentialLR

# --------------------- 随机种子 & 设备 ---------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# --------------------- 全局超参（优化版） ---------------------
BATCH_SIZE = 128
EPOCHS = 150
LEARNING_RATE = 0.0008
WEIGHT_DECAY = 8e-4
LABEL_SMOOTHING = 0.25
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 20
WARMUP_EPOCHS = 15

# --------------------- 数据集加载 ---------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_file = os.path.join(script_dir, "RML2016.10a_dict.pkl")
with open(pkl_file, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# RML2016.10a 的 11 类调制
mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
    'CPFSK', 'GFSK', 'PAM4', 'AM-SSB', 'AM-DSB', 'WBFM']
mod2idx = {m: i for i, m in enumerate(mods)}
num_classes = len(mods)
print(f"分类类别总数：{num_classes}")

X, y, snr_mod_labels = [], [], []
for (mod, snr), sigs in data.items():
    if mod in mod2idx:
        sample_num = len(sigs)
        X.append(sigs)
        y.extend([mod2idx[mod]] * sample_num)
        # 调制+SNR联合标签，双重分层划分
        snr_mod_labels.extend([f"{mod}_{snr}"] * sample_num)

X = np.vstack(X).astype(np.float32)      # (N, 2, 128)
y = np.array(y).astype(np.int64)
print(f"原始数据集样本数: {X.shape[0]}")

# ===================== 4通道时频融合特征 =====================
X_complex = X[:, 0, :] + 1j * X[:, 1, :]
X_fft = np.fft.fftshift(np.fft.fft(X_complex, axis=1), axes=1)
# 对数幅度压缩动态范围
amp = np.abs(X_fft)
X_log_amp = np.log10(amp + 1e-8)
# 相位解缠绕消除2π跳变
X_unwrap_phase = np.unwrap(np.angle(X_fft), axis=1)
# 原始时域I/Q
I = X[:, 0, :]
Q = X[:, 1, :]
# 拼接4通道 [I, Q, 对数幅度, 解缠相位]
X = np.stack([I, Q, X_log_amp, X_unwrap_phase], axis=1)
print(f"特征维度转换完成，shape: {X.shape} (4通道时频融合)")

# --------------------- 双重分层划分数据集 ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=snr_mod_labels
)

# --------------------- 各通道独立归一化 ---------------------
eps = 1e-6
train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
train_std = np.std(X_train, axis=(0, 2), keepdims=True)
X_train = (X_train - train_mean) / (train_std + eps)
X_test = (X_test - train_mean) / (train_std + eps)

# --------------------- 增强射频在线数据增强Dataset ---------------------
class RFAugDataset(Dataset):
    def __init__(self, data, label, train=True):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)
        self.train = train

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx].clone()
        label = self.label[idx]
        if self.train:
            # 1. 更强高斯噪声扰动
            noise = torch.randn_like(x) * 0.08
            x = x + noise
            # 2. I/Q/幅度通道随机缩放
            scale = torch.FloatTensor(1).uniform_(0.75, 1.25)
            x[[0, 1, 2], :] = x[[0, 1, 2], :] * scale
            # 3. 相位通道大范围偏移
            phase_shift = torch.FloatTensor(1).uniform_(-0.3, 0.3)
            x[3, :] = x[3, :] + phase_shift
            # 4. 频谱随机滚动模拟频偏
            shift = torch.randint(-6, 6, ()).item()
            x = torch.roll(x, shifts=shift, dims=-1)
        return x, label

train_dataset = RFAugDataset(X_train, y_train, train=True)
test_dataset = RFAugDataset(X_test, y_test, train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

# --------------------- SE通道注意力模块 ---------------------
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

# --------------------- 带注意力、Dropout、GELU残差块 ---------------------
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
        # 短路连接
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

# --------------------- 改进1D ResNet网络（适配4通道输入） ---------------------
class ResNet1D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 预卷积stride=1，减缓下采样，保留频谱细节
        self.pre_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        # 三层深层残差组
        self.layer1 = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 256),
            ResidualBlock1D(256, 256)
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

# --------------------- 训练初始化（类别加权损失） ---------------------
model = ResNet1D(num_classes=num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 类别权重：CPFSK(5)、GFSK(6)、AM-SSB(8)、AM-DSB(9)加大权重
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.2, 2.2, 1.0, 1.8, 1.8, 1.0]).to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, weight=class_weights)

# 学习率调度：15轮预热 + 余弦退火重启
warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=WARMUP_EPOCHS)
cos_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_scheduler], milestones=[WARMUP_EPOCHS])

writer = SummaryWriter()
best_acc = 0.0
best_epoch = 0
patience_counter = 0

# --------------------- 训练/验证函数 ---------------------
def train_epoch(loader, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(pred, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/train", acc, epoch)
    return avg_loss, acc

def evaluate(loader, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = loss_fn(pred, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(pred, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    writer.add_scalar("Loss/test", avg_loss, epoch)
    writer.add_scalar("Acc/test", acc, epoch)
    return avg_loss, acc

# --------------------- 主训练循环（带早停） ---------------------
print("开始训练...")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(train_loader, epoch)
    test_loss, test_acc = evaluate(test_loader, epoch)
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
          f"LR: {current_lr:.2e}")

    # 保存最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(script_dir, "best_rf_resnet_optim3.pt"))
        patience_counter = 0
        print(f"✅ 保存新最优模型，当前最佳准确率：{best_acc:.4f}")
    else:
        patience_counter += 1
        # 早停判断
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 早停触发：连续{EARLY_STOP_PATIENCE}轮精度无提升，终止训练")
            break

print(f"\n训练完成！最佳测试准确率: {best_acc:.4f} (Epoch {best_epoch})")
print("最优模型已保存为 best_rf_resnet_optim3 .pt")
writer.close()