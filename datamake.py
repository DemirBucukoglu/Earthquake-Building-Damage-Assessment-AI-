import os
import csv
import shutil
import random

# ----------- CONFIGURATION -----------
# Your patchwork folder path, which contains the 500 mosaic images and mosaic_annotations.csv.
PATCHWORK_DIR = r"C:\Users\demir\OneDrive\Belgeler\CS CLUB\patchwork"

# Output directory for the YOLOv5-style dataset. It will be created inside the PATCHWORK_DIR.
OUTPUT_DIR = os.path.join(PATCHWORK_DIR, "patchwork_yolo")

# Train/validation split ratio (e.g., 80% training and 20% validation).
train_ratio = 0.8

# Class mapping: "damaged" is mapped to 0, "undamaged" to 1.
class_map = {
    "damaged": 0,
    "undamaged": 1
}
# -------------------------------------

def clean_folder(folder):
    """Delete the folder if it exists and then recreate it."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

def main():
    # Define output subdirectories.
    images_train_dir = os.path.join(OUTPUT_DIR, "images", "train")
    images_val_dir   = os.path.join(OUTPUT_DIR, "images", "val")
    labels_train_dir = os.path.join(OUTPUT_DIR, "labels", "train")
    labels_val_dir   = os.path.join(OUTPUT_DIR, "labels", "val")
    
    # Clean up existing output directories to avoid accumulating old files.
    clean_folder(os.path.join(OUTPUT_DIR, "images"))
    clean_folder(os.path.join(OUTPUT_DIR, "labels"))
    
    # Recreate the individual subdirectories.
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    
    # Path to the CSV file.
    csv_path = os.path.join(PATCHWORK_DIR, "mosaic_annotations.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Build a dictionary where each key is an image filename and the value is a list of bounding boxes.
    # Each bounding box is stored as a tuple: (img_width, img_height, class, xmin, ymin, xmax, ymax)
    annotations_dict = {}
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row["filename"].strip()  # Strip extra spaces if any.
            img_width = float(row["width"])
            img_height = float(row["height"])
            cls = row["class"].strip()
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])
            
            if filename not in annotations_dict:
                annotations_dict[filename] = []
            annotations_dict[filename].append((img_width, img_height, cls, xmin, ymin, xmax, ymax))
    
    # Get the unique filenames from the CSV.
    all_filenames = list(annotations_dict.keys())
    print("Total unique filenames in CSV:", len(all_filenames))  # Should be 500.
    
    # Shuffle and split into training and validation.
    random.shuffle(all_filenames)
    num_train = int(len(all_filenames) * train_ratio)
    train_filenames = set(all_filenames[:num_train])
    val_filenames   = set(all_filenames[num_train:])
    
    print("Number of training images:", len(train_filenames))
    print("Number of validation images:", len(val_filenames))
    
    # Process each image and corresponding annotations.
    for filename, bbox_list in annotations_dict.items():
        # Construct the source image path from PATCHWORK_DIR.
        src_img_path = os.path.join(PATCHWORK_DIR, filename)
        if not os.path.exists(src_img_path):
            print(f"Warning: Image file does not exist: {src_img_path}")
            continue
        
        # Determine whether the image goes to training or validation.
        if filename in train_filenames:
            img_out_dir = images_train_dir
            label_out_dir = labels_train_dir
        else:
            img_out_dir = images_val_dir
            label_out_dir = labels_val_dir
        
        # Copy the image to its output folder.
        shutil.copy2(src_img_path, os.path.join(img_out_dir, filename))
        
        # Create the YOLO-format annotation file.
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_file_path = os.path.join(label_out_dir, label_filename)
        yolo_lines = []
        
        for (img_w, img_h, cls_str, xmin, ymin, xmax, ymax) in bbox_list:
            class_id = class_map.get(cls_str, -1)
            if class_id == -1:
                print(f"Skipping annotation for unknown class: {cls_str} in {filename}")
                continue
            
            # Convert the bounding box from absolute coordinates to normalized YOLO format.
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            w_norm = (xmax - xmin) / img_w
            h_norm = (ymax - ymin) / img_h
            
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write out the .txt file.
        with open(label_file_path, 'w') as label_file:
            label_file.write("\n".join(yolo_lines) + "\n")
    
    print("\nConversion complete! YOLO-style dataset created in:")
    print(OUTPUT_DIR)
    print("images/train :", len(os.listdir(images_train_dir)), "files")
    print("images/val   :", len(os.listdir(images_val_dir)), "files")
    print("labels/train :", len(os.listdir(labels_train_dir)), "files")
    print("labels/val   :", len(os.listdir(labels_val_dir)), "files")

if __name__ == "__main__":
    main()
