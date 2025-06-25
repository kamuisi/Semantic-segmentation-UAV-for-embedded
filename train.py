import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

import os
from glob import glob

from PIL import Image

from fast_scnn import Fast_SCNN

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')

class LoveDA_Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
        self.to_tensor = ToTensor()
    def __getitem__(self, index):
        image = self.to_tensor(self.images[index])
        mask = torch.tensor(self.masks[index], dtype=torch.long)

        return image, mask
    def __len__(self):
        return len(self.images)

def train(model, dataloader, device, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, epochs, power=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0

        for img, mask in dataloader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            # print(torch.unique(mask))

            loss = criterion(output, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")
    # if isinstance(model, torch.nn.DataParallel):
    #   torch.save(model.module.state_dict(), "fast_scnn_model.pth")
    # else:
    #   torch.save(model.state_dict(), "fast_scnn_model.pth")
    torch.save(model.state_dict(), "fast_scnn_model.pth")
    print("✅ Model saved as `fast_scnn_model.pth`")

# ------------------------ Main ------------------------
def run_training_pipeline(image_dir, mask_dir, num_classes=8, batch_size=4, epochs=100):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {DEVICE}")

    # Load all images and masks
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
    assert len(image_paths) == len(mask_paths), "Mismatch between image and mask count!"

    images, masks = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        images.append(img)
        masks.append(mask)

    dataset = LoveDA_Dataset(images, masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Fast_SCNN(num_classes=num_classes).to(DEVICE)
    # if torch.cuda.device_count() > 1:
    #   model = nn.DataParallel(model)
    train(model, dataloader, DEVICE, epochs=epochs)

if __name__ == "__main__":
    set_seed(2025)
    run_training_pipeline(
        image_dir="./images_png",
        mask_dir="./masks_png",
        num_classes=8,
        batch_size=24,
        epochs=200
    )
    