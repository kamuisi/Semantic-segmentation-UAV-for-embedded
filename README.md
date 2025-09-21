
# UAV Semantic Segmentation on Embedded System

This is a mini project from my **Artificial Intelligence for Embedded System** course. It demonstrates how to run an AI model on an embedded system with limited resources.


## Features

- Achieves **48 FPS** with **0.257 F1** score on an Intel i5-1135G7 CPU
- Runs at **0.5 FPS** with **0.042 F1** score on a Raspberry Pi Zero 2 W
## Installation

Core training dependencies
```bash
pip install torch torchvision numpy pillow tqdm
```

Export and quantization tools **(Linux system required)**
```bash
pip install --upgrade onnx onnxscript torch numpy onnx2tf onnx_graphsurgeon sng4onnx tf_keras
```

For testing ONNX and TFLite models
```bash
pip install onnxruntime tensorflow
```

Optional: for visualization
```bash
pip install matplotlib
```
    
## Usage
After installation, move the dataset to the specific folders: **images_png** and **masks_png**.

Start training with: `python train.py` (use `-h` to show arguments)

Test trained model with: `python test_model.py`

For model export and quantization **(Linux system required)**: `python3 export_model.py`

Test exported models:
- ONNX: `python check_middle_export.py`
- TFLite: `python check_final_export.py`

## Dataset
This project uses the [LoveDA dataset](https://zenodo.org/records/5706578), which provides semantic segmentation data for urban and rural scenes.
## Documentation

- [Fast_SCNN](https://arxiv.org/abs/1902.04502)

