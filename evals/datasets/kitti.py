"""
KITTI depth dataset for probe training.

Expects the standard KITTI Depth Prediction Benchmark layout:
    <raw_root>/<date>/<date>_drive_XXXX_sync/image_02/data/<frame>.png
    <label_root>/{train,val}/<date>_drive_XXXX_sync/proj_depth/groundtruth/image_02/<frame>.png
"""
import os

import numpy as np
import torch
from PIL import Image

from .utils import get_kitti_transforms


def KITTI(
    raw_root,
    label_root,
    split,
    image_mean="imagenet",
    augment_train=True,
    frame_stride=10,
    image_size=(352, 1216),
):
    assert split in ["train", "trainval", "valid", "test"]
    if split in ["valid", "test"]:
        label_subdir = "val"
        # full eval: no subsampling on the validation set
        stride = 1
        augment = False
    else:
        label_subdir = "train"
        stride = frame_stride
        augment = augment_train and split == "train"

    return KITTI_Depth(
        raw_root=raw_root,
        label_root=label_root,
        label_subdir=label_subdir,
        split=split,
        image_mean=image_mean,
        augment=augment,
        frame_stride=stride,
        image_size=image_size,
    )


class KITTI_Depth(torch.utils.data.Dataset):
    name = "KITTI"
    max_depth = 80.0
    min_depth = 1e-3

    def __init__(
        self,
        raw_root,
        label_root,
        label_subdir,
        split,
        image_mean,
        augment,
        frame_stride,
        image_size,
    ):
        super().__init__()
        self.raw_root = raw_root
        self.label_root = label_root
        self.image_size = image_size  # (H, W) of the bottom-center crop

        self.image_transform, self.shared_transform = get_kitti_transforms(
            image_mean,
            image_size,
            augment,
            additional_targets={"depth": "image"},
        )

        self.samples = self._build_index(
            os.path.join(label_root, label_subdir), raw_root, frame_stride
        )
        print(f"KITTI {split}: {len(self.samples)} frames "
              f"(stride={frame_stride}, label={label_subdir})")

    @staticmethod
    def _build_index(label_dir, raw_root, frame_stride):
        samples = []
        for drive in sorted(os.listdir(label_dir)):
            depth_dir = os.path.join(
                label_dir, drive, "proj_depth", "groundtruth", "image_02"
            )
            if not os.path.isdir(depth_dir):
                continue
            date = drive[:10]
            rgb_dir = os.path.join(raw_root, date, drive, "image_02", "data")
            for fname in sorted(os.listdir(depth_dir)):
                if not fname.endswith(".png"):
                    continue
                rgb_path = os.path.join(rgb_dir, fname)
                if not os.path.exists(rgb_path):
                    continue
                samples.append((rgb_path, os.path.join(depth_dir, fname)))
        return samples[::frame_stride]

    def __len__(self):
        return len(self.samples)

    def _bottom_center_crop(self, image, depth):
        """Crop to self.image_size from the bottom-center of the frame.

        KITTI native sizes vary slightly per drive (1224-1242 wide, 370-376
        high). Cropping (vs. resizing) preserves the camera intrinsics, and
        the bottom anchor keeps road geometry while discarding sky -- this is
        the BTS / NeWCRFs / AdaBins convention for KITTI depth.
        """
        h_target, w_target = self.image_size
        h, w = image.shape[:2]
        assert h >= h_target and w >= w_target, (
            f"frame {h}x{w} smaller than crop {h_target}x{w_target}"
        )
        top = h - h_target
        left = (w - w_target) // 2
        image = image[top:top + h_target, left:left + w_target]
        depth = depth[top:top + h_target, left:left + w_target]
        return image, depth

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        image = np.array(Image.open(rgb_path).convert("RGB"))  # uint8 H,W,3
        depth = np.array(Image.open(depth_path), dtype=np.int32).astype(np.float32) / 256.0

        # community-standard valid mask: zero anything outside [min_depth, max_depth]
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0

        image, depth = self._bottom_center_crop(image, depth)

        image = self.image_transform(image)  # (3, H, W) float
        if self.shared_transform is not None:
            transformed = self.shared_transform(
                image=image.permute(1, 2, 0).numpy(),
                depth=depth[:, :, None],
            )
            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
        else:
            depth = torch.tensor(depth).float()[None, :, :]

        return {"image": image, "depth": depth}
