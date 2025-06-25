import onnxruntime as ort
import numpy as np
from PIL import Image
import glob
import os


ort_session = ort.InferenceSession("fast_scnn.onnx")

image_dir = "./Rural/images_png/"
mask_dir = "./Rural/masks_png/"
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

correct_pixels = 0
total_pixels = 0

for img_path, mask_path in zip(image_paths, mask_paths):

    img = Image.open(img_path).convert("RGB").resize((320, 320))
    arr = np.array(img).astype(np.float32) / 255.0 
    arr = np.transpose(arr, (2, 0, 1))              
    arr = np.expand_dims(arr, axis=0)               

    
    onnx_inputs = {ort_session.get_inputs()[0].name: arr}
    output = ort_session.run(None, onnx_inputs)[0]  

    pred = np.argmax(output, axis=1).squeeze(0)     

    
    mask = np.array(Image.open(mask_path).resize((320, 320)), dtype=np.int64)

    correct_pixels += (pred == mask).sum()
    total_pixels += mask.size

accuracy = correct_pixels / total_pixels
print(f"ONNX Pixel Accuracy: {accuracy:.4f}")