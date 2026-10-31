import pickle
import numpy as np
import os

# ===================== 你只需要改这里 =====================
# 你的文件名：RML2016.10a.dat 或者 RML2016.10b.dat
dat_filename = "RML2016.10a_dict.dat"  
# =========================================================

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
dat_filename = os.path.join(script_dir, dat_filename)

print("正在把 .dat 转成 .pkl...")

# 官方格式：(I/Q两路, 128点采样, 调制类型, SNR)
# 该文件是Python2下用cPickle保存的，需要指定encoding=latin-1或bytes
with open(dat_filename, "rb") as f:
    data = pickle.load(f, encoding="latin-1")

# 转成数组格式
for key in data:
    data[key] = np.array(data[key])

# 保存为 PKL
pkl_name = dat_filename.replace(".dat", ".pkl")
with open(pkl_name, "wb") as f:
    pickle.dump(data, f)

print(f"转换完成！生成文件：{pkl_name}")
print("现在你可以用这个 pkl 训练模型、生成 best.pt 了！")