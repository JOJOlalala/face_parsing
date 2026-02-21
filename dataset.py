from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from palette import rgb_to_label

IMG_EXTS = {".png", ".jpg", ".jpeg"}


class FaceParsingDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, file_list=None, augment=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.augment = augment

        # collect image paths
        if file_list is None:
            self.img_paths = sorted(
                [
                    p
                    for p in self.img_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                ]
            )
        else:
            self.img_paths = [self.img_dir / fn for fn in file_list]

        # build mask lookup by stem
        if self.mask_dir is not None:
            self.mask_map = {
                p.stem: p
                for p in self.mask_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            }
            missing = [p.name for p in self.img_paths if p.stem not in self.mask_map]
            if missing:
                raise FileNotFoundError(
                    f"Missing masks for {len(missing)} files, e.g. {missing[:5]}"
                )

        print(f"Loaded {len(self.img_paths)} samples from {self.img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        if self.mask_dir is not None:
            mask_path = self.mask_map[img_path.stem]
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
            mask = rgb_to_label(mask_rgb)
        else:
            mask = None

        if self.augment is not None and mask is not None:
            img, mask = self.augment(img, mask)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if mask is not None:
            mask = torch.from_numpy(mask).long()
            return img, mask, img_path.name
        else:
            return img, img_path.name
