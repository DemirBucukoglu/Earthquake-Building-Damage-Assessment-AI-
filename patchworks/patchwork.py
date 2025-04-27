import os
import cv2
import numpy as np
import random
import math
import csv


damaged_dir = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\archive\Earthquake\Damaged Building"
undamaged_dir = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\archive\Earthquake\Undamaged Building"


output_dir = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\patchwork"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

csv_path = os.path.join(output_dir, "mosaic_annotations.csv")

allowed_extensions = ('.jpg', '.jpeg', '.png')
patch_w, patch_h = 224, 224    
group_size = 10  # number of patches per mosaic                
n_cols = 5       # fixed number of columns in each mosaic


damaged_images = [os.path.join(damaged_dir, f) for f in os.listdir(damaged_dir)
                  if f.lower().endswith(allowed_extensions)]
undamaged_images = [os.path.join(undamaged_dir, f) for f in os.listdir(undamaged_dir)
                    if f.lower().endswith(allowed_extensions)]

data = []
for img_path in damaged_images:
    data.append((img_path, "damaged"))
for img_path in undamaged_images:
    data.append((img_path, "undamaged"))

# Set the number of mosaics to generate.
num_mosaics = 500
print("Total number of mosaics to be created:", num_mosaics)

with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])

    for mosaic_index in range(num_mosaics):
        
        group_data = random.choices(data, k=group_size)
        n_images = len(group_data)
        n_rows = math.ceil(n_images / n_cols)

        mosaic_width = n_cols * patch_w
        mosaic_height = n_rows * patch_h
        mosaic_img = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

        bboxes = []
        labels = []
        
        for idx, (img_path, label) in enumerate(group_data):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Resize patch.
            img_resized = cv2.resize(img, (patch_w, patch_h))

            # Place image in grid.
            row = idx // n_cols
            col = idx % n_cols
            x1 = col * patch_w
            y1 = row * patch_h
            x2 = x1 + patch_w
            y2 = y1 + patch_h

            mosaic_img[y1:y2, x1:x2] = img_resized

            bboxes.append((x1, y1, x2, y2))
            labels.append(label)

        # Annotate each patch with a bounding box and the respective label.
        for (x1, y1, x2, y2), label in zip(bboxes, labels):
            color = (0, 0, 255) if label == "damaged" else (0, 255, 0)
            cv2.rectangle(mosaic_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(mosaic_img, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        mosaic_filename = f"mosaic_{mosaic_index + 1}.jpg"
        mosaic_filepath = os.path.join(output_dir, mosaic_filename)
        cv2.imwrite(mosaic_filepath, mosaic_img)
        print(f"Saved {mosaic_filepath}")

        # Write annotations for each patch in the CSV.
        for (x1, y1, x2, y2), label in zip(bboxes, labels):
            writer.writerow([
                mosaic_filename,  # mosaic filename
                mosaic_width,     # width of the mosaic image
                mosaic_height,    # height of the mosaic image
                label,            # damaged or undamaged
                x1,               # xmin
                y1,               # ymin
                x2,               # xmax
                y2                # ymax
            ])

print("Mosaic creation and CSV annotations generation complete!")
