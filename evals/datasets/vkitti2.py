import os
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from transform import Resize, NormalizeImage, PrepareForNet, Crop

def VKITTI2(
    train_path,
    test_path,
    split
):
    assert split in ["train", "valid", "trainval", "test"]

    if split == "test":
        return VKITTI2_data(test_path)
    else:
        return VKITTI2_data(train_path)

        
class VKITTI2_data(Dataset):
    def __init__(self, filelist_path, size=(518, 518)):
        
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])]))
    
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m

        min_depth = 0.001
        max_depth = 80.0
        depth[(depth < min_depth) | (depth > max_depth)] = 0
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth']).float()[None, :, :]
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)

        
