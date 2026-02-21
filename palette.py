import numpy as np
from PIL import Image

COLOR_LIST = np.array([
    [0, 0, 0],        # 0 background
    [204, 0, 0],      # 1 skin
    [76, 153, 0],     # 2 nose
    [204, 204, 0],    # 3 eye_g
    [51, 51, 255],    # 4 l_eye
    [204, 0, 204],    # 5 r_eye
    [0, 255, 255],    # 6 l_brow
    [255, 204, 204],  # 7 r_brow
    [102, 51, 0],     # 8 l_ear
    [255, 0, 0],      # 9 r_ear
    [102, 204, 0],    # 10 mouth
    [255, 255, 0],    # 11 u_lip
    [0, 0, 153],      # 12 l_lip
    [0, 0, 204],      # 13 hair
    [255, 51, 153],   # 14 hat
    [0, 204, 204],    # 15 ear_r
    [0, 51, 0],       # 16 neck_l
    [255, 153, 51],   # 17 neck
    [0, 204, 0],      # 18 cloth
], dtype=np.uint8)

NUM_CLASSES = len(COLOR_LIST)

def rgb_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    mask_rgb = mask_rgb.astype(np.uint8)
    packed = (mask_rgb[..., 0].astype(np.int32) << 16) | (mask_rgb[..., 1].astype(np.int32) << 8) | mask_rgb[..., 2].astype(np.int32)
    color_packed = (COLOR_LIST[:, 0].astype(np.int32) << 16) | (COLOR_LIST[:, 1].astype(np.int32) << 8) | COLOR_LIST[:, 2].astype(np.int32)

    lut = {int(c): int(i) for i, c in enumerate(color_packed)}
    label = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    unknown = np.ones(mask_rgb.shape[:2], dtype=bool)

    for c, i in lut.items():
        m = (packed == c)
        label[m] = i
        unknown[m] = False

    if unknown.any():
        unk_vals = np.unique(mask_rgb[unknown].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown colors found: {unk_vals[:10]} (up to 10 shown).")

    return label

def label_to_rgb(label: np.ndarray) -> np.ndarray:
    label = label.astype(np.int64)
    if label.min() < 0 or label.max() >= NUM_CLASSES:
        raise ValueError(f"Label out of range: min={label.min()}, max={label.max()}, num_classes={NUM_CLASSES}")
    return COLOR_LIST[label]

def save_label_as_rgb_png(label_hw: np.ndarray, path: str):
    rgb = label_to_rgb(label_hw)
    Image.fromarray(rgb).save(path)
