import torch
from torch import nn
import torch.nn.functional as F
from fast_scnn import Fast_SCNN
import numpy as np
import subprocess

class FakeAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        in_h, in_w = x.shape[2], x.shape[3]
        out_h, out_w = self.output_size

        stride_h = in_h // out_h
        stride_w = in_w // out_w

        kernel_h = in_h - (out_h - 1) * stride_h
        kernel_w = in_w - (out_w - 1) * stride_w

        return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))

class PyramidPoolingModuleExport(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 4
        self.stages = nn.ModuleList([
            nn.Sequential(
                FakeAdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                FakeAdaptiveAvgPool2d((2, 2)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                FakeAdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                FakeAdaptiveAvgPool2d((6, 6)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            out.append(upsampled)
        out = torch.cat(out, dim=1)
        out = self.bottleneck(out)
        return out
    
def copy_bn_stats(src_module, dst_module):
    for src, dst in zip(src_module.modules(), dst_module.modules()):
        if isinstance(src, nn.BatchNorm2d) and isinstance(dst, nn.BatchNorm2d):
            dst.running_mean.data = src.running_mean.data.clone()
            dst.running_var.data = src.running_var.data.clone()
            dst.num_batches_tracked.data = src.num_batches_tracked.data.clone()

def export_onnx():
    model = Fast_SCNN(num_classes=8)
    model.load_state_dict(torch.load("fast_scnn_model.pth", map_location='cpu'))
    model.eval()
    export_ppm = PyramidPoolingModuleExport(128)
    copy_bn_stats(model.gobal_feature[-1], export_ppm)
    model.gobal_feature[-1] = export_ppm  # đổi PyramidPoolingModule dể export
    dummy_input = (torch.randn(1, 3, 320, 320),)
    torch.onnx.export(model, dummy_input, "fast_scnn.onnx")

def onnx_to_tf():
    subprocess.run(["onnx2tf", "-i", "fast_scnn.onnx", "-o", "tf_model", "-oiqt"])

if __name__ == "__main__":
    export_onnx()
    onnx_to_tf()
    print("🎉 All done! Your TFLite model is ready.")