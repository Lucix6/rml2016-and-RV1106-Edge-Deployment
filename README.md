# rml2016-and-RV1106-Edge-Deployment
将rml2016a数据集训练成模型,部署到luckyfoxpicomax rv1106 上


请前往 RadioML 官网下载 RML2016.10a_dict.pkl，放置在项目根目录。

运行 python dump_test_set.py，提取出测试集并归一化，生成 X_test.npy 和 y_test.npy。

运行 python convert_to_bin.py，生成板端推理所需的 test_data.bin 和 test_labels.bin。

运行 python test_SNR_pkl.py，生成用于计算不同信噪比准确率的 test_snrs.bin。

将生成的 3 个 .bin 文件通过 adb push 推送到开发板 /root/ 目录。


