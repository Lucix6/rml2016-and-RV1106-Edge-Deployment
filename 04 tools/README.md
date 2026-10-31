第一步：生成 .npy 文件

脚本：dump_test_set.py（或者类似脚本）

输入：RML2016.10a_dict.pkl（原始数据集）

输出：X_test.npy 和 y_test.npy

第二步：生成模型推理用的 .bin 文件

脚本：convert_to_bin.py（之前你删掉了，上传 GitHub 时记得把它加回来，放到 04_tools/ 目录）

输入：X_test.npy 和 y_test.npy

输出：

test_data.bin（纯 float32 数据）

test_labels.bin（纯 int8 标签）

第三步：生成测试 SNR 的 .bin 文件

脚本：test_SNR_pkl.py（在 Windows 目录里）

输入：RML2016.10a_dict.pkl

输出：test_snrs.bin（纯 int8 的 SNR 数组）
