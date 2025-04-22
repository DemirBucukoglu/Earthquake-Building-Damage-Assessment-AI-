"""
pathwork3.0.py
-------------------------------------------------
Run from any folder with:
  python pathwork3.0.py --n 2500 --train_ratio 0.8 \
    --damaged_dir <path> --undamaged_dir <path> --out_root <path>
Creates n synthetic 1024×1024 scenes plus YOLO labels,
split into train/val sets based on --train_ratio.
"""

import cv2
import os
import glob
import random
import argparse
import numpy as np


def build_synthetic_scenes(n, train_ratio, damaged_dir, undamaged_dir, out_root):
    # Create train/val directories
    img_train_dir = os.path.join(out_root, "images", "train")
    img_val_dir   = os.path.join(out_root, "images", "val")
    lbl_train_dir = os.path.join(out_root, "labels", "train")
    lbl_val_dir   = os.path.join(out_root, "labels", "val")
    for d in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir):
        os.makedirs(d, exist_ok=True)
    
    # Load source crops
    exts = ("*.jpg", "*.jpeg", "*.png")
    damaged   = [f for e in exts for f in glob.glob(os.path.join(damaged_dir, e))]
    undamaged = [f for e in exts for f in glob.glob(os.path.join(undamaged_dir, e))]
    assert damaged and undamaged, "Source directories must contain images."

    print(f"[INFO] {len(damaged)} damaged + {len(undamaged)} undamaged crops loaded")

    W = H = 1024
    for idx in range(n):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        labels = []
        k = random.randint(5, 15)
        for _ in range(k):
            cls = random.randint(0, 1)
            fname = random.choice(damaged if cls == 0 else undamaged)
            crop = cv2.imread(fname)
            if crop is None:
                continue

            # random scale and optional rotation
            scale = random.uniform(0.30, 0.60)
            crop  = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if random.random() < 0.3:
                rot = random.choice([cv2.ROTATE_90_CLOCKWISE,
                                     cv2.ROTATE_180,
                                     cv2.ROTATE_90_COUNTERCLOCKWISE])
                crop = cv2.rotate(crop, rot)

            h, w = crop.shape[:2]
            if h >= H or w >= W:
                continue

            # random placement
            x0 = random.randint(0, W - w - 1)
            y0 = random.randint(0, H - h - 1)
            canvas[y0:y0 + h, x0:x0 + w] = crop

            # YOLO label
            xc = (x0 + w / 2) / W
            yc = (y0 + h / 2) / H
            labels.append(f"{cls} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")

        # split into train or val
        if random.random() < train_ratio:
            img_dir = img_train_dir
            lbl_dir = lbl_train_dir
        else:
            img_dir = img_val_dir
            lbl_dir = lbl_val_dir

        out_name = f"{idx:06d}"
        cv2.imwrite(os.path.join(img_dir, out_name + ".jpg"), canvas)
        with open(os.path.join(lbl_dir, out_name + ".txt"), "w") as f:
            f.write("\n".join(labels))

        if (idx + 1) % 100 == 0:
            print(f"[INFO] {idx + 1}/{n} scenes done")

    print("[DONE] synthetic dataset written to", out_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500, help="# of synthetic scenes")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Proportion for training set")
    ap.add_argument("--damaged_dir", required=True, help="Path to damaged building crops")
    ap.add_argument("--undamaged_dir", required=True, help="Path to undamaged building crops")
    ap.add_argument("--out_root", required=True, help="Root output folder for images/labels")
    args = ap.parse_args()

    build_synthetic_scenes(
        n=args.n,
        train_ratio=args.train_ratio,
        damaged_dir=args.damaged_dir,
        undamaged_dir=args.undamaged_dir,
        out_root=args.out_root
    )
