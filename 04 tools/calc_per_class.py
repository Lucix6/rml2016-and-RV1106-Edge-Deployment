import numpy as np
from pathlib import Path

# 1. 类别名称
class_names = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 
               'CPFSK', 'GFSK', 'PAM4', 'AM-SSB', 'AM-DSB', 'WBFM']

ROOT = Path(__file__).resolve().parent

# 2. 读取真实标签和预测结果
y_true = np.load(ROOT / "y_test.npy").astype(int)
y_pred = np.loadtxt(ROOT / "pred_int8.txt", dtype=int)

if len(y_true) != len(y_pred):
    print(f"❌ 长度不匹配！y_true: {len(y_true)}, y_pred: {len(y_pred)}")
    exit()

print(f"总共测试样本数: {len(y_true)}\n")
print(f"🌟 整体准确率 (Overall Accuracy): {np.mean(y_true == y_pred):.4f}\n")
print("-" * 65)
print(f"{'类别名称':<10} | {'单类准确率':<10} | {'正确数 / 总数':<15} | {'预测为该类的总数'}")
print("-" * 65)

# 3. 纯 Numpy 计算每一类的指标
for i, name in enumerate(class_names):
    # 真实标签为该类的所有样本
    true_mask = (y_true == i)
    total_true = np.sum(true_mask)
    
    # 其中预测正确的数量
    correct_pred = np.sum(y_pred[true_mask] == i)
    
    # 该类准确率 (Recall)
    acc = correct_pred / total_true if total_true > 0 else 0.0
    
    # 预测为该类的总数 (用于看误报)
    total_pred = np.sum(y_pred == i)
    
    print(f"{name:<12} | {acc:<12.4f} | {correct_pred:>4d} / {total_true:<6d} | {total_pred:>6d}")

print("-" * 65)

# 4. 纯 Numpy 生成混淆矩阵
cm = np.zeros((11, 11), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

print("\n🧩 混淆矩阵 (行: 真实标签, 列: 预测标签):")
# 打印表头
header = " " * 6 + "".join([f"{n:>7}" for n in class_names])
print(header)
# 打印每一行
for i, name in enumerate(class_names):
    row = f"{name:<6}" + "".join([f"{val:>7d}" for val in cm[i]])
    print(row)