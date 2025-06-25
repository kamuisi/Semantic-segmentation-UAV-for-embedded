import torch
from fast_scnn import Fast_SCNN
import subprocess
# import tensorflow as tf

def export_to_onnx():
    model = Fast_SCNN(num_classes=8)
    model.load_state_dict(torch.load("fast_scnn_model.pth", map_location='cpu'))
    model.eval()

    sample_input = (torch.randn(1, 3, 320, 320),)
    torch.onnx.export(model, sample_input, "fast_scnn.onnx")

def onnx_to_tf():
    subprocess.run(["onnx2tf", "-i", "fast_scnn.onnx", "-o", "tf_model", "-oiqt"])

# def tf_to_tflite():
#     converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
#     converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#     tflite_model = converter.convert()
#     with open("fast_scnn.tflite", "wb") as f:
#         f.write(tflite_model)
if __name__ == '__main__':
    export_to_onnx()
    onnx_to_tf()