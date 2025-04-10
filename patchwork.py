import os
import cv2
import numpy as np
import random
import math
import csv

# --- Input and Output Directories ---

# Input directories for damaged and undamaged building images.
damaged_dir = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\archive\Earthquake\Damaged Building"
undamaged_dir = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\archive\Earthquake\Undamaged Building"

# Output directory to save mosaic images.
output_dir = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\patchwork"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

csv_path = os.path.join(output_dir, "mosaic_annotations.csv")

allowed_extensions = ('.jpg', '.jpeg', '.png')
patch_w, patch_h = 224, 224    
group_size = 10                
n_cols = 5                   

damaged_images = [os.path.join(damaged_dir, f) for f in os.listdir(damaged_dir)
                  if f.lower().endswith(allowed_extensions)]
undamaged_images = [os.path.join(undamaged_dir, f) for f in os.listdir(undamaged_dir)
                    if f.lower().endswith(allowed_extensions)]

data = []
for img_path in damaged_images:
    data.append((img_path, "damaged"))
for img_path in undamaged_images:
    data.append((img_path, "undamaged"))

random.shuffle(data)

num_groups = math.ceil(len(data) / group_size)
print("Total number of mosaics to be created:", num_groups)

with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])

    for group_index in range(num_groups):
        group_data = data[group_index * group_size: (group_index + 1) * group_size]
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

            img_resized = cv2.resize(img, (patch_w, patch_h))

            row = idx // n_cols
            col = idx % n_cols
            x1 = col * patch_w
            y1 = row * patch_h
            x2 = x1 + patch_w
            y2 = y1 + patch_h

            mosaic_img[y1:y2, x1:x2] = img_resized

            bboxes.append((x1, y1, x2, y2))
            labels.append(label)

        for (x1, y1, x2, y2), label in zip(bboxes, labels):
            color = (0, 0, 255) if label == "damaged" else (0, 255, 0)
            cv2.rectangle(mosaic_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(mosaic_img, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        mosaic_filename = f"mosaic_{group_index + 1}.jpg"
        mosaic_filepath = os.path.join(output_dir, mosaic_filename)
        cv2.imwrite(mosaic_filepath, mosaic_img)
        print(f"Saved {mosaic_filepath}")

        # Write annotation for each patch into the CSV.
        for (x1, y1, x2, y2), label in zip(bboxes, labels):

            writer.writerow([
                mosaic_filename,  
                mosaic_width,     
                mosaic_height,    
                label,            
                x1,               
                y1,               
                x2,               
                y2                
            ])

print("Mosaic creation and CSV annotations generation complete!")

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md