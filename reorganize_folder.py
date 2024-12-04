import os
from shutil import copy2
from PIL import Image
import numpy as np

# Apply binary threshold and calculate the backgroud pixel ratio
def filter_images_by_black_ratio(image_dir, output_dir, binary_threshold=200, black_ratio_threshold=0.7):
    """
    Filters images based on the ratio of black pixels after applying a binary threshold.

    Args:
        image_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where filtered images will be saved.
        binary_threshold (int): Threshold for converting grayscale to binary (default: 200).
        black_ratio_threshold (float): Minimum black pixel ratio required to pass the filter (default: 0.7).
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('L')  # Convert image to grayscale
            image_array = np.array(image)
            
            # Apply binary threshold
            binary_image = (image_array < binary_threshold).astype(np.uint8)
            black_ratio = np.sum(binary_image) / binary_image.size  # Calculate black pixel ratio
            
            # Copy images that pass the black ratio threshold
            if black_ratio >= black_ratio_threshold:
                copy2(image_path, output_dir)

# Determine label based on mask tumor pixel ratio
def generate_label_from_mask(mask_path, threshold=0.1):
    """
    Generates a label ('tumor' or 'benign') based on the ratio of white pixels in the mask.

    Args:
        mask_path (str): Path to the mask image.
        threshold (float): Minimum white pixel ratio to classify as 'tumor' (default: 0.1).

    Returns:
        str: 'tumor' if white pixel ratio >= threshold, else 'benign'.
    """
    mask = np.array(Image.open(mask_path).convert('L'))
    white_pixel_count = np.sum(mask == 255)
    total_pixels = mask.size  # Total number of pixels in the mask
    white_ratio = white_pixel_count / total_pixels
    return "tumor" if white_ratio >= threshold else "benign"

# Organize images into 'tumor' and 'benign' directories based on mask classification
def organize_images_by_mask(image_dir, mask_dir, output_dir, threshold=0.1):
    """
    Organizes images into 'tumor' and 'benign' directories based on corresponding mask labels.

    Args:
        image_dir (str): Path to the directory containing input images.
        mask_dir (str): Path to the directory containing mask images.
        output_dir (str): Path to the directory where organized images will be saved.
        threshold (float): Minimum white pixel ratio in the mask to classify as 'tumor' (default: 0.1).
    """
    os.makedirs(output_dir, exist_ok=True)
    tumor_dir = os.path.join(output_dir, "tumor")
    benign_dir = os.path.join(output_dir, "benign")
    os.makedirs(tumor_dir, exist_ok=True)
    os.makedirs(benign_dir, exist_ok=True)

    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith(".png"):
            mask_path = os.path.join(mask_dir, mask_file)
            label = generate_label_from_mask(mask_path, threshold)

            # Assume image and mask files share the same base name
            image_file = mask_file.replace(".png", ".jpg")
            image_path = os.path.join(image_dir, image_file)

            # Copy images to appropriate directories based on their labels
            if os.path.exists(image_path):
                if label == "tumor":
                    copy2(image_path, tumor_dir)
                else:
                    copy2(image_path, benign_dir)

# Train set processing
# Step 1: Filter images based on black pixel ratio
image_dir = "/home/wooju.chung/Havard_TMA_samples/images/train_cases"  # Original image folder
filtered_output_dir = "/home/wooju.chung/TMA_filtered_images/train"  # Folder for filtered images
filter_images_by_black_ratio(image_dir, filtered_output_dir, binary_threshold=200, black_ratio_threshold=0.7)

# Step 2: Organize filtered images based on mask labels
mask_dir = "/home/wooju.chung/Havard_TMA_samples/masks/train_cases" # Mask folder
output_dir = "/home/wooju.chung/organized_TMA2/train" # Folder for organized dataset
organize_images_by_mask(filtered_output_dir, mask_dir, output_dir)

# Validation set processing
# Step 1
image_dir = "/home/wooju.chung/Havard_TMA_samples/images/val_cases"
filtered_output_dir = "/home/wooju.chung/TMA_filtered_images/val"
filter_images_by_black_ratio(image_dir, filtered_output_dir, binary_threshold=200, black_ratio_threshold=0.7)

# Step 2
mask_dir = "/home/wooju.chung/Havard_TMA_samples/masks/val_cases"
output_dir = "/home/wooju.chung/organized_TMA2/val"
organize_images_by_mask(filtered_output_dir, mask_dir, output_dir)
