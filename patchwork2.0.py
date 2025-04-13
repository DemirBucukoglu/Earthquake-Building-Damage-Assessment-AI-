"""
build_synthetic_scenes.py
-------------------------------------------------
Run from any folder with:
  python build_synthetic_scenes.py --n 2500
Creates n synthetic 1024×1024 scenes plus YOLO labels.
"""

import cv2, os, glob, random, argparse
import numpy as np

# --------------------------------------------------------------------------- #
# 1. command‑line arguments
# --------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2500, help="# synthetic scenes")
args = ap.parse_args()

# --------------------------------------------------------------------------- #
# 2. user paths  (edit if you move folders)
# --------------------------------------------------------------------------- #
damaged_dir   = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\archive\Earthquake\Damaged Building"
undamaged_dir = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\archive\Earthquake\Undamaged Building"
out_root      = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\pathcwork2.0"

out_img_dir   = os.path.join(out_root, "images")
out_lbl_dir   = os.path.join(out_root, "labels")
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

# --------------------------------------------------------------------------- #
# 3. collect source crops
# --------------------------------------------------------------------------- #
exts = ("*.jpg", "*.jpeg", "*.png")
damaged   = [f for e in exts for f in glob.glob(os.path.join(damaged_dir, e))]
undamaged = [f for e in exts for f in glob.glob(os.path.join(undamaged_dir, e))]
assert damaged and undamaged, "Could not find source images!"

print(f"[INFO] {len(damaged)} damaged + {len(undamaged)} undamaged crops loaded")

# helper to pick a class & image
def sample_crop():
    cls = random.randint(0, 1)                    # 0 damaged, 1 undamaged
    fname = random.choice(damaged if cls == 0 else undamaged)
    img = cv2.imread(fname)
    return cls, img

# --------------------------------------------------------------------------- #
# 4. build scenes
# --------------------------------------------------------------------------- #
W = H = 1024
for idx in range(args.n):
    canvas  = np.zeros((H, W, 3), dtype=np.uint8)          # black background
    labels  = []
    k       = random.randint(5, 15)                        # roofs per scene

    for _ in range(k):
        cls, crop = sample_crop()
        if crop is None:
            continue

        # --- random scale (shrink) & optional rotation ---------------------
        scale = random.uniform(0.30, 0.60)
        crop  = cv2.resize(crop, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        if random.random() < 0.3:                          # 30 % rotate 0/90/180/270
            rot = random.choice([cv2.ROTATE_90_CLOCKWISE,
                                 cv2.ROTATE_180,
                                 cv2.ROTATE_90_COUNTERCLOCKWISE])
            crop = cv2.rotate(crop, rot)

        h, w   = crop.shape[:2]
        if h >= H or w >= W:               # skip if still too large
            continue

        # --- random position (must fit) ------------------------------------
        x0 = random.randint(0, W - w - 1)
        y0 = random.randint(0, H - h - 1)
        canvas[y0:y0 + h, x0:x0 + w] = crop

        # --- YOLO label (normalised) ---------------------------------------
        xc = (x0 + w / 2) / W
        yc = (y0 + h / 2) / H
        labels.append(f"{cls} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")

    # ------------------------------------------------------------------ #
    # 5. save
    # ------------------------------------------------------------------ #
    out_name = f"{idx:06d}"
    cv2.imwrite(os.path.join(out_img_dir,  out_name + ".jpg"), canvas)
    with open(os.path.join(out_lbl_dir, out_name + ".txt"), "w") as f:
        f.write("\n".join(labels))

    if (idx + 1) % 100 == 0:
        print(f"[INFO] {idx + 1}/{args.n} scenes done")

print("[DONE] synthetic dataset written to", out_root)
