import numpy as np
import os
import cv2
from collections import Counter
from tqdm import tqdm

def compute_class_frequencies(mask_dir, num_classes=151, background_class=255):
    """
    Compute the frequency of each class in the ADE20K segmentation dataset, including the background class.

    Args:
        mask_dir (str): Path to the directory containing segmentation masks.
        num_classes (int): Number of foreground classes in the dataset (0 to num_classes-1).
        background_class (int): Pixel value representing the background class.

    Returns:
        dict: A dictionary with class indices (0 to num_classes-1 and background) as keys and their frequencies as values.
    """
    class_counts = Counter()
    total_pixels = 0

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Flatten mask and count occurrences of each class including background
        unique, counts = np.unique(mask, return_counts=True)
        class_counts.update(dict(zip(unique, counts)))

        # Update total pixel count
        total_pixels += mask.size

    # Compute normalized frequency (including background class)
    class_frequencies = {cls: count for cls, count in class_counts.items()}

    return class_frequencies

# Example usage
mask_dir = "../data/ADEChallengeData2016/annotations/training"  # Update with actual path
class_frequencies = compute_class_frequencies(mask_dir)

# Print class frequencies
for cls, freq in sorted(class_frequencies.items()):
    print(f"Class {cls}: {freq:.6f}")
