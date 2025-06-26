# **About our project**
In this project we used Fast_SCNN to build a semantic segmentation model with [LoveDA dataset](https://zenodo.org/records/5706578). 
Beside that we applied knownlegde learned from **AI for Embedded courses** to perform model quantization and deployment on a Raspberry Pi Zero 2W.
Our model, after training, achieves ~48FPS and 0.62 accuracy on laptop CPU. 
After exporting and quantizing to TFLITE, its run at ~0.5FPS with 0.0245 accuracy on Raspberry Pi.
# **Fast_SCNN reference**
Our architecture based on the origin [Fast_SCNN paper](https://arxiv.org/abs/1902.04502).

 
 
 
 
 
 
 
 
