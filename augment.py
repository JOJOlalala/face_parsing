import albumentations as A
import numpy as np
import cv2


def get_train_augment():
    """
    Augmentations tuned for face parsing:
    - keep geometry mild (tiny parts are sensitive)
    - allow scale/shift/rotate and photometric jitter
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # mild geometric aug
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.20,
                rotate_limit=10,
                border_mode=0,  # constant
                value=(0, 0, 0),
                mask_value=0,
                p=0.7,
            ),
            # photometric aug (image only)
            A.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05, p=0.6
            ),
            # A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            # # optional: small cutout (don’t overdo for segmentation)
            # A.CoarseDropout(
            #     max_holes=6,
            #     max_height=32,
            #     max_width=32,
            #     min_holes=1,
            #     fill_value=0,
            #     mask_fill_value=0,
            #     p=0.15,
            # ),
        ]
    )


def get_train_augment_safe():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02, p=0.5
            ),
            # A.ShiftScaleRotate(
            #     shift_limit=0.02,
            #     scale_limit=0.05,
            #     rotate_limit=5,
            #     border_mode=cv2.BORDER_REFLECT_101,
            #     value=(0, 0, 0),
            #     mask_value=0,
            #     p=0.3,
            # ),
        ]
    )


def get_best_augment():
    """
    No augmentation — intentional for face parsing with 19 asymmetric classes.

    Why augmentation typically hurts here:

    1. ColorJitter: face parsing uses color as a discriminative feature
       (skin tone, lip color, hair colour). Jittering these cues directly
       degrades the signal the model relies on.

    2. HorizontalFlip WITHOUT label swapping: when the image flips,
       l_eye ends up on the right side of the face but still carries the
       l_eye label. The model then sees contradictory spatial priors
       (l_eye sometimes left, sometimes right). Correct flipping requires
       swapping paired labels: l_eye↔r_eye, l_brow↔r_brow, l_ear↔r_ear.

    3. GaussianBlur: fine boundaries (eyelids, lip edges) need sharp
       gradients. Blurring those boundaries during training makes the
       model learn softer, less precise boundaries.

    4. ShiftScaleRotate: fills border pixels with mask_value=0 (background),
       corrupting face boundary labels at every augmented crop edge.

    Returns an identity transform (no-op) so aug_fn can still be passed
    to the dataset without conditional logic.
    """
    return A.Compose([])  # identity — no augmentation


# ---------------------------------------------------------------------------
# Correct horizontal flip for face parsing with asymmetric label pairs
# ---------------------------------------------------------------------------

# Classes that must be swapped when the image is mirrored left↔right.
# eye_g(3), mouth(10), u_lip(11), l_lip(12), nose(2), hair(13), skin(1),
# hat(14), ear_r(15=earring), neck_l(16=necklace), neck(17), cloth(18)
# are either symmetric or accessories with no left/right counterpart.
FLIP_LABEL_PAIRS = [
    (4, 5),   # l_eye  ↔ r_eye
    (6, 7),   # l_brow ↔ r_brow
    (8, 9),   # l_ear  ↔ r_ear
]


def flip_with_label_swap(
    img: np.ndarray,
    mask: np.ndarray,
    p: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Horizontally flip image and mask with probability p, then swap
    laterally-paired labels so spatial semantics stay consistent.

    Without swapping, the model sees contradictory training examples:
      non-flipped → left side of face has l_eye label
      flipped     → left side of face has r_eye label  (wrong)
    With swapping both are consistent: l_eye always refers to the
    face's own left eye regardless of which way the image is oriented.

    Args:
        img:  (H, W, 3) uint8 image
        mask: (H, W)    int64 label map with values 0-18
        p:    probability of applying the flip

    Returns:
        (img, mask) — either original or flipped+swapped copies
    """
    if np.random.random() >= p:
        return img, mask

    img_out  = img[:, ::-1, :].copy()   # flip image
    mask_out = mask[:, ::-1].copy()     # flip mask spatially

    # swap paired labels
    for label_a, label_b in FLIP_LABEL_PAIRS:
        mask_a = mask_out == label_a
        mask_b = mask_out == label_b
        mask_out[mask_a] = label_b
        mask_out[mask_b] = label_a

    return img_out, mask_out


def make_face_aug(
    p_flip: float = 0.5,
    p_geom: float = 0.7,
    p_color: float = 0.7,
    p_blur: float = 0.15,
) -> callable:
    """
    Combined augmentation for face parsing that correctly handles label swapping.

    Pipeline:
      1. flip_with_label_swap  — spatially correct, swaps l/r paired labels
      2. ShiftScaleRotate      — mild geometric distortion
      3. ColorJitter           — photometric variation
      4. GaussianBlur          — optional light blur

    The albumentations pipeline is built WITHOUT HorizontalFlip so the
    label-aware flip in step 1 is the only horizontal flip applied.
    """
    album = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.20,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=p_geom,
        ),
        A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.15, hue=0.05, p=p_color),
        A.GaussianBlur(blur_limit=(3, 5), p=p_blur),
    ])

    def aug_fn(img: np.ndarray, mask: np.ndarray):
        img, mask = flip_with_label_swap(img, mask, p=p_flip)
        out = album(image=img, mask=mask.astype(np.int32))
        return out["image"], out["mask"].astype(np.int64)

    return aug_fn


def apply_aug(aug, img_uint8: np.ndarray, mask_int64: np.ndarray):
    """
    img_uint8: (H,W,3) uint8
    mask_int64: (H,W) int labels 0..18
    """
    out = aug(image=img_uint8, mask=mask_int64.astype(np.int32))
    img2 = out["image"]
    mask2 = out["mask"].astype(np.int64)
    return img2, mask2
