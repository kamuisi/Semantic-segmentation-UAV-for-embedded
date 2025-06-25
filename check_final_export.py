import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="fast_scnn_full_integer_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lấy danh sách ảnh và mask
image_dir = "Urban/images_png"
mask_dir = "Urban/masks_png"
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

correct_pixels = 0
total_pixels = 0

for img_path, mask_path in zip(image_paths, mask_paths):
    # Load và preprocess ảnh
    img = Image.open(img_path).convert("RGB").resize((320, 320))
    arr = np.array(img).astype(np.float32) / 255.0  # (320, 320, 3)
    arr = np.transpose(arr, (2, 0, 1))  # (3, 320, 320)
    arr = np.expand_dims(arr, axis=0)   # (1, 3, 320, 320)

    # Nếu model TFLite nhận (1, 320, 320, 3), cần chuyển lại
    if list(input_details[0]['shape']) == [1, 320, 320, 3]:
        arr = np.transpose(arr, (0, 2, 3, 1))  # (1, 320, 320, 3)

    # Nếu model int8/uint8, scale lại
    if input_details[0]['dtype'] == np.uint8:
        arr = (arr * 255).astype(np.uint8)
    elif input_details[0]['dtype'] == np.int8:
        arr = (arr * 255 - 128).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])  # (1, num_classes, 320, 320) hoặc (1, 320, 320, num_classes)

    # Nếu output là (1, 320, 320, num_classes), chuyển về (1, num_classes, 320, 320)
    if output.shape[-1] == output.shape[1] and output.shape[1] != 3:
        output = np.transpose(output, (0, 3, 1, 2))
        
    pred = np.argmax(output, axis=-1).squeeze(0)  # (320, 320)

    # Load mask
    mask = np.array(Image.open(mask_path).resize((320, 320)), dtype=np.int64)
    if mask.ndim == 3:
        mask = mask[:, :, 0]  # Nếu mask là ảnh màu

    correct_pixels += (pred == mask).sum()
    total_pixels += mask.size

accuracy = correct_pixels / total_pixels
print(f"TFLite Pixel Accuracy: {accuracy:.4f}")