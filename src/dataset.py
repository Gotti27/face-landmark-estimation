import numpy as np
from torch.utils import data
import torch
from PIL import Image
import torchvision
import pandas as pd
import os


class WFLW(data.Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations_file, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = str(os.path.join(self.img_dir, self.annotations.iloc[idx, -1]))
        image = torchvision.io.decode_image(img_path)

        landmarks = torch.from_numpy(self.annotations.iloc[idx, 0: 98 * 2].values.astype(np.float32))
        bounding_box = [
            self.annotations.iloc[idx, 98 * 2: (98 * 2) + 4].values.tolist()[:2],
            self.annotations.iloc[idx, 98 * 2: (98 * 2) + 4].values.tolist()[2:]
        ]
        attributes = self.annotations.iloc[idx, (98 * 2) + 4: (98 * 2) + 4 + 6].values.tolist()

        top_left = bounding_box[0][1] - 10
        top_right = bounding_box[1][1] + 10
        bottom_left = bounding_box[0][0] - 10
        bottom_right = bounding_box[1][0] + 10

        top_left = top_left if top_left > 0 else 0
        bottom_left = bottom_left if bottom_left > 0 else 0

        image = image[:, top_left:top_right, bottom_left:bottom_right]

        scale_x = 224 / image.shape[1]
        scale_y = 224 / image.shape[2]

        landmarks[1::2] -= top_left
        landmarks[1::2] *= scale_x

        landmarks[::2] -= bottom_left
        landmarks[::2] *= scale_y

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            landmarks = self.target_transform(landmarks)

        return image, landmarks, bounding_box, attributes
