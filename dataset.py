from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from palette import rgb_to_label

IMG_EXTS = {".png", ".jpg", ".jpeg"}


class FaceParsingDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, file_list=None, augment=None, cache=False):
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

        # Optional in-memory cache: pre-decode all images and masks at init.
        # Eliminates per-sample PIL I/O + rgb_to_label on every __getitem__.
        # Cost: ~750 MB RAM for 900×512×512 images+masks (masks stored as uint8).
        self._img_cache  = None
        self._mask_cache = None
        if cache:
            self._img_cache  = []
            self._mask_cache = [] if self.mask_dir is not None else None
            for p in self.img_paths:
                self._img_cache.append(
                    np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
                )
                if self.mask_dir is not None:
                    mask_rgb = np.array(
                        Image.open(self.mask_map[p.stem]).convert("RGB"), dtype=np.uint8
                    )
                    self._mask_cache.append(rgb_to_label(mask_rgb).astype(np.uint8))
            print(f"  → cached {len(self._img_cache)} decoded arrays in RAM")

        print(f"Loaded {len(self.img_paths)} samples from {self.img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        if self._img_cache is not None:
            img  = self._img_cache[idx].copy()
            mask = self._mask_cache[idx].astype(np.int64) if self._mask_cache is not None else None
        else:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            if self.mask_dir is not None:
                mask_rgb = np.array(Image.open(self.mask_map[img_path.stem]).convert("RGB"), dtype=np.uint8)
                mask = rgb_to_label(mask_rgb)
            else:
                mask = None

        if self.augment is not None and mask is not None:
            img, mask = self.augment(img, mask)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if mask is not None:
            mask = torch.from_numpy(np.asarray(mask, dtype=np.int64))
            return img, mask, img_path.name
        else:
            return img, img_path.name
