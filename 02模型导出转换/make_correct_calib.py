import pickle
import gc
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "calib_fixed"
CALIB_DIR.mkdir(exist_ok=True)

# 1. 读取数据集
class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

with (ROOT / "RML2016.10a_dict.pkl").open("rb") as f:
    data = CompatibleUnpickler(f, encoding="latin1").load()

mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'CPFSK', 'GFSK', 'PAM4', 'AM-SSB', 'AM-DSB', 'WBFM']
mod2idx = {m: i for i, m in enumerate(mods)}
X, y, strat_labels = [], [], []
for (mod, snr), sigs in data.items():
    if mod in mod2idx:
        X.append(sigs); y.extend([mod2idx[mod]] * len(sigs)); strat_labels.extend([f"{mod}_{snr}"] * len(sigs))

X = np.vstack(X).astype(np.float32)
y = np.array(y).astype(np.int64)

# 2. FFT 特征提取
X_complex = X[:, 0, :] + 1j * X[:, 1, :]
X_fft = np.fft.fftshift(np.fft.fft(X_complex, axis=1), axes=1)
X_log_amp = np.log10(np.abs(X_fft) + 1e-8)
X_unwrap_phase = np.unwrap(np.angle(X_fft), axis=1)
X = np.stack([X[:, 0, :], X[:, 1, :], X_log_amp, X_unwrap_phase], axis=1)

# 3. 划分并求归一化参数
X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat_labels)
eps = 1e-6
train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
train_std  = np.std(X_train,  axis=(0, 2), keepdims=True)

# 4. 抽取 200 个样本，归一化后保存
indices = np.random.default_rng(42).choice(len(X_train), 200, replace=False)
paths = []
for i, idx in enumerate(indices):
    sample = (X_train[idx] - train_mean) / (train_std + eps)
    p = CALIB_DIR / f"sample_{i:03d}.npy"
    np.save(p, sample.astype(np.float32))
    paths.append(p.resolve())

# 5. 生成新的 dataset_fixed.txt
with (ROOT / "dataset_fixed.txt").open("w") as f:
    for p in paths: f.write(f"{p}\n")

print("✅ 正确预处理的校准集已生成: dataset_fixed.txt 和 calib_fixed/")