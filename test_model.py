import torch
from fast_scnn import Fast_SCNN
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os

import argparse

paser = argparse.ArgumentParser()

paser.add_argument("-d", "--direction", help="Duong dan den tap anh", type=str, default="./Rural/")
paser.add_argument("-m", "--model", help="Ten model", type=str, default="fast_scnn_model.pth")

arg = paser.parse_args()

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Fast_SCNN(num_classes=8)
    model.load_state_dict(torch.load(arg.model, map_location=device))
    model.eval()
    model.to(device=device)

    images = []
    masks = []
    image_paths = sorted(glob(os.path.join(arg.direction+"images_png/", "*.png")))
    mask_paths = sorted(glob(os.path.join(arg.direction+"masks_png/", "*.png")))
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        images.append(img)
        masks.append(mask)

    # colors = [
    #     (0, 0, 0),
    #     (255, 255, 255),
    #     (255, 0, 0),
    #     (255, 255, 0),
    #     (0, 0, 255),
    #     (159, 129, 183),
    #     (0, 255, 0),
    #     (255, 195, 128)
    # ]

    to_tensor = transforms.ToTensor()

    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for image, mask in zip(images, masks):
            input_tensor = to_tensor(image).unsqueeze(0).to(device)
            output = model(input_tensor)     
            probs = torch.softmax(output, dim=1)              
            prediction = probs.argmax(dim=1).squeeze(0)
            prediction_np = prediction.cpu().numpy()  

            correct_pixels += (prediction_np == mask).sum()
            total_pixels += mask.size
    
    accuracy = correct_pixels / total_pixels
    print(f"Accuracy: {accuracy:.4f}")
            # segmented_image = np.zeros((prediction_np.shape[0], prediction_np.shape[1], 3), dtype=np.uint8)
            # colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

            # for class_id, color in enumerate(colors):
            #     segmented_image[prediction == class_id] = color
            #     colored_mask[mask == class_id] = color

            # plt.figure(figsize=(12, 5))
            # plt.subplot(1, 3, 1)
            # plt.title("Input Image")
            # plt.imshow(image)

            # plt.subplot(1, 3, 2)
            # plt.title("Predicted Segmentation")
            # plt.imshow(segmented_image)

            # plt.subplot(1, 3, 3)
            # plt.title("Expected label")
            # plt.imshow(colored_mask)
            # plt.show()

if __name__ == "__main__":
    test()