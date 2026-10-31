from pathlib import Path
from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent
ONNX_DIR = ROOT.parent / "ONNX"
ONNX = ONNX_DIR / "modulation_resnet_4d.onnx"
OUT_RKNN = ONNX_DIR / "modulation_resnet_4d_rv1106.rknn"
DATASET = ONNX_DIR / "dataset.txt"

rknn = RKNN(verbose=True)
rknn.config(
    target_platform="rv1106",
    mean_values=[[0, 0, 0, 0]],
    std_values=[[1, 1, 1, 1]],
    quantized_dtype="asymmetric_quantized-8",
    quantized_algorithm="normal"
)
rknn.load_onnx(model=str(ONNX), inputs=["features"], input_size_list=[[1, 4, 128]])
rknn.build(do_quantization=True, dataset=str(DATASET))
rknn.export_rknn(str(OUT_RKNN))
rknn.release()
print(f"✅ RKNN 导出成功: {OUT_RKNN}")
