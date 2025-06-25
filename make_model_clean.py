import torch
from collections import OrderedDict

state_dict = torch.load("fast_scnn_model.pth", map_location="cpu")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
torch.save(new_state_dict, "fast_scnn_model_clean.pth")