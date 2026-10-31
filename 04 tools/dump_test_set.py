#!/usr/bin/env python3
"""内存友好的 dump_test_set.py"""
import pickle
import gc
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent

class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
        'CPFSK', 'GFSK', 'PAM4', 'AM-SSB', 'AM-DSB', 'WBFM']
mod2idx = {m: i for i, m in enumerate(mods)}

with (ROOT / "RML2016.10a_dict.pkl").open("rb") as f:
    data = CompatibleUnpickler(f, encoding="latin1").load()

# ---------- 逐条处理, 先算 4 通道特征, 释放中间变量 ----------
feat_list, y_list, strat_labels = [], [], []
for (mod, snr), sigs in data.items():
    if mod not in mod2idx:
        continue
    # sigs: (n, 2, 128) -> 转 float32
    s = np.asarray(sigs, dtype=np.float32)          # (n, 2, 128)
    i_ch = s[:, 0, :]                               # (n, 128)
    q_ch = s[:, 1, :]                               # (n, 128)

    # 复数用 complex64 省一半内存
    c = (i_ch + 1j * q_ch).astype(np.complex64)     # (n, 128)
    fft_c = np.fft.fftshift(np.fft.fft(c, axis=1), axes=1)

    log_amp = np.log10(np.abs(fft_c) + 1e-8).astype(np.float32)
    unwrap_ph = np.unwrap(np.angle(fft_c), axis=1).astype(np.float32)

    feat = np.stack([i_ch, q_ch, log_amp, unwrap_ph], axis=1)  # (n, 4, 128)
    feat_list.append(feat)
    y_list.append(np.full(len(s), mod2idx[mod], dtype=np.int64))
    strat_labels.extend([f"{mod}_{snr}"] * len(s))

    del s, c, fft_c, log_amp, unwrap_ph, feat
    gc.collect()

X = np.concatenate(feat_list, axis=0)
y = np.concatenate(y_list, axis=0)
del feat_list, y_list
gc.collect()
print("4 通道特征:", X.shape)

# ---------- 划分 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=strat_labels
)
del X, y, strat_labels
gc.collect()

# ---------- 归一化 ----------
eps = 1e-6
train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
train_std  = np.std(X_train,  axis=(0, 2), keepdims=True)
X_test = (X_test - train_mean) / (train_std + eps)

np.save(ROOT / "X_test.npy", X_test.astype(np.float32))
np.save(ROOT / "y_test.npy", y_test)
print("已保存 X_test.npy:", X_test.shape, "y_test.npy:", y_test.shape)
print("每类样本数:", np.bincount(y_test, minlength=len(mods)).tolist())
