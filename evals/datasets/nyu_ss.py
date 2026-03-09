import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import get_nyu_transforms

def NYUSemantic(
    root_path,  
    split,
    image_mean="imagenet",
    center_crop=False,
    rotateflip=False,
    augment_train=False,
    **kwargs
):

    assert split in ["train", "test", "valid", "trainval"]
    return NYUSemanticDataset(
        path=root_path,
        split=split,
        image_mean=image_mean,
        center_crop=center_crop,
        augment_train=augment_train,
        rotateflip=rotateflip,
    )


class NYUSemanticDataset(Dataset):
    def __init__(
        self,
        path,
        split,
        image_mean="imagenet",
        center_crop=False,
        augment_train=False,
        rotateflip=False,
    ):
        super().__init__()
        self.name = "NYUv2_Semantic"
        self.center_crop = center_crop
        self.root_dir = path

        augment = augment_train and ("train" in split)
        image_size = (480, 480) if center_crop else (480, 640)

        self.image_transform, self.shared_transform = get_nyu_transforms(
            image_mean=image_mean,
            image_size=image_size,
            augment=augment,
            rotateflip=rotateflip,
            additional_targets={"mask": "mask"}, 
        )

        split_file_name = "train" if "train" in split else "test"
        split_file = os.path.join(path, f"{split_file_name}.txt")
        
        with open(split_file, "r") as f:
            self.file_names = f.read().splitlines()

        print(f"NYU-Semantic {split}: {len(self.file_names)} instances loaded.")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        name = self.file_names[index]
        
        img_path = os.path.join(self.root_dir, "image", f"{name}.png")
        mask_path = os.path.join(self.root_dir, "label40", f"{name}.png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        image = self.image_transform(image)
        
        if self.center_crop:
            image = image[..., 80:-80]
            mask = mask[..., 80:-80]

        if self.shared_transform:
            image = image.permute(1, 2, 0).numpy()
            transformed = self.shared_transform(image=image, mask=mask)
            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            mask = torch.tensor(transformed["mask"]).long()
        else:
            mask = torch.tensor(mask).long()


        return {
            "image": image,
            "mask": mask,
            "name": name,  
        }