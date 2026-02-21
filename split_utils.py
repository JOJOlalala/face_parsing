from pathlib import Path
import random

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(img_dir: str):
    p = Path(img_dir)
    files = [
        x.name for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS
    ]
    files.sort()
    return files


def make_split(files: list[str], val_ratio: float = 0.1, seed: int = 42):
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_files = files[:n_val]
    train_files = files[n_val:]
    return train_files, val_files


def save_split(
    train_files: list[str],
    val_files: list[str],
    out_dir: str = "splits",
    tag: str = "seed42",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / f"train_{tag}.txt"
    val_path = out / f"val_{tag}.txt"
    train_path.write_text("\n".join(train_files) + "\n")
    val_path.write_text("\n".join(val_files) + "\n")
    return str(train_path), str(val_path)


def load_list(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
