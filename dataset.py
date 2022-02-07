
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

def get_labels_info(info_path):
    info = pd.read_csv(info_path)
    # info has format: [['obstacles' 59 193 246]...]
    # info = info.to_numpy()
    class_names = np.array(info["name"])
    labels_values = np.array(info[["r","g","b"]])
    return class_names, labels_values

def convert_data(img, label, info_path):
    img = img/255.0
    class_names, labels_values = get_labels_info(info_path)
    sematic_maps = []
    for color in labels_values:
        same = np.equal(label, color)
        class_map = np.all(same,axis=-1)
        sematic_maps.append(class_map)
    semantic_map = np.array(np.stack(sematic_maps,axis=-1))
    return img, semantic_map



class RC_dataset(Dataset):
    def __init__(self, image_dir, label_dir, info_path, transform = None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.info_path = info_path

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.image_dir,self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".png","_converted.png"))
        img = np.array(Image.open(img_path).convert("RGB"),dtype=np.float32)
        label = np.array(Image.open(label_path),dtype=np.float32)

        img, label = convert_data(img, label, self.info_path)
        if self.transform is not None:
            augmentation = self.transform(image=img,mask=label)
            img = augmentation["image"]
            label = augmentation["mask"]

        return img, label

