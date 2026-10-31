import onnx
from onnx import helper, numpy_helper
import numpy as np
from pathlib import Path

# 🌟 关键修改：自动获取当前脚本所在的目录
ROOT = Path(__file__).resolve().parent

# 拼接绝对路径
input_path = ROOT / "D:\\myfile\\study_file\\脚本文件\\deep-learning\\ONNX\\modulation_resnet.onnx"
output_path = ROOT / "D:\\myfile\\study_file\\脚本文件\\deep-learning\\ONNX\\modulation_resnet_4d.onnx"

# 1. 加载你现有的 11 类模型
model = onnx.load(str(input_path))

# 原输出名是 "logits"
old_output_name = "logits"

# 2. 创建一个常量张量，定义目标形状 [1, 11, 1, 1]
shape_tensor = numpy_helper.from_array(
    np.array([1, 11, 1, 1], dtype=np.int64), 
    name="reshape_shape"
)
model.graph.initializer.append(shape_tensor)

# 3. 创建一个 Reshape 节点
# 输入是旧的 logits，输出取个新名字叫 logits_4d
reshape_node = helper.make_node(
    "Reshape",
    inputs=[old_output_name, "reshape_shape"],
    outputs=["logits_4d"]
)
model.graph.node.append(reshape_node)

# 4. 把原图的输出信息替换为 4D 输出信息
# 清除原来的输出（因为原来的 logits 现在已经变成了内部张量）
model.graph.output.pop(0)

# 添加新的输出
new_output = helper.make_tensor_value_info(
    "logits_4d", 
    onnx.TensorProto.FLOAT, 
    [1, 11, 1, 1]
)
model.graph.output.append(new_output)

# 5. 检查并保存
onnx.checker.check_model(model)
onnx.save(model, str(output_path))
print(f"✅ 成功生成 4D 输出模型: {output_path}")