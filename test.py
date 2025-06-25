import torch
import torchvision.models as models

mobilenet_v3 = models.mobilenet_v2(weights='DEFAULT')

x = torch.randn(1, 3, 1024, 1024)
out = x
for i, layer in enumerate(mobilenet_v3.features):
    out = layer(out)
    print(f"Layer {i} output shape: {out.shape}")