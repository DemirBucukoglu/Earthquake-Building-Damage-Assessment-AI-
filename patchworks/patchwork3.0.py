"""
patchwork3.0.py
-------------------------------------------------
Run from any folder with:
  python patchwork3.0.py --n 2500 --train_ratio 0.8 \
    --damaged_dir <path> --undamaged_dir <path> --out_root <path>
Creates n synthetic 1024×1024 scenes plus YOLO labels,
split into train/val sets under a new top‑level folder `patchwork3.0`.
"""

import cv2
import os
import glob
import random
import argparse
import numpy as np


def build_synthetic_scenes(n, train_ratio, damaged_dir, undamaged_dir, out_root):
    # create patchwork3.0 folder
    base_root = os.path.join(out_root, 'patchwork3.0')
    # image and label directories
    img_train_dir = os.path.join(base_root, 'images', 'train')
    img_val_dir   = os.path.join(base_root, 'images', 'val')
    lbl_train_dir = os.path.join(base_root, 'labels', 'train')
    lbl_val_dir   = os.path.join(base_root, 'labels', 'val')
    for d in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir):
        os.makedirs(d, exist_ok=True)

    # load source crops
    exts = ('*.jpg', '*.jpeg', '*.png')
    damaged = [f for e in exts for f in glob.glob(os.path.join(damaged_dir, e))]
    undamaged = [f for e in exts for f in glob.glob(os.path.join(undamaged_dir, e))]
    assert damaged and undamaged, 'Could not find source images!'
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

            # random scale & rotation
            scale = random.uniform(0.30, 0.60)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if random.random() < 0.3:
                rot = random.choice([cv2.ROTATE_90_CLOCKWISE,
                                     cv2.ROTATE_180,
                                     cv2.ROTATE_90_COUNTERCLOCKWISE])
                crop = cv2.rotate(crop, rot)

            h, w = crop.shape[:2]
            if h >= H or w >= W:
                continue

            # place crop
            x0 = random.randint(0, W - w - 1)
            y0 = random.randint(0, H - h - 1)
            canvas[y0:y0 + h, x0:x0 + w] = crop

            # YOLO format label
            xc = (x0 + w / 2) / W
            yc = (y0 + h / 2) / H
            labels.append(f"{cls} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")

        # choose split
        if random.random() < train_ratio:
            img_dir, lbl_dir = img_train_dir, lbl_train_dir
        else:
            img_dir, lbl_dir = img_val_dir, lbl_val_dir

        name = f"{idx:06d}"
        cv2.imwrite(os.path.join(img_dir, name + '.jpg'), canvas)
        with open(os.path.join(lbl_dir, name + '.txt'), 'w') as f:
            f.write("\n".join(labels))

        if (idx + 1) % 100 == 0:
            print(f"[INFO] {idx + 1}/{n} scenes done")

    print(f"[DONE] synthetic dataset written to {base_root}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2500, help='# scenes')
    ap.add_argument('--train_ratio', type=float, default=0.8, help='train set ratio')
    ap.add_argument('--damaged_dir', required=True, help='path to damaged images')
    ap.add_argument('--undamaged_dir', required=True, help='path to undamaged images')
    ap.add_argument('--out_root', required=True, help='base output path')
    args = ap.parse_args()

    build_synthetic_scenes(
        args.n,
        args.train_ratio,
        args.damaged_dir,
        args.undamaged_dir,
        args.out_root
    )
