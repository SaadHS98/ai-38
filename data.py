import os
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn.model_selection import KFold

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Load the image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Resize the image to 512x512 using OpenCV
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        # Convert the mask to grayscale (1-channel) and standardize values to 0 and 1
        mask[mask == 255] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

