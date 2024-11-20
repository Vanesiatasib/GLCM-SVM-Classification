import os
import numpy as np
from PIL import Image
import cv2

def generate_histogram(img):
    """Generates a normalized histogram for an image."""
    histogram, _ = np.histogram(img.flatten(), bins=256, range=(0, 255))
    return histogram / histogram.sum()

def match_histogram(input_img, ref_hist):
    """Matches the histogram of an input image to a reference histogram."""
    input_hist, _ = np.histogram(input_img.flatten(), bins=256, range=(0, 255))
    input_cdf = np.cumsum(input_hist) / input_img.size  # Input image CDF
    ref_cdf = np.cumsum(ref_hist)                      # Reference histogram CDF

    pixel_map = np.searchsorted(ref_cdf, input_cdf)
    matched_img = pixel_map[input_img]
    return matched_img

# Define folder paths
ng_folder = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\2_GrayScale\NG"
ok_folder = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\2_GrayScale\OK"
ng_output_folder = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\3_Histogram\NG"
ok_output_folder = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\3_Histogram\OK"

# Load reference image and compute its histogram
ref_img_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\02_Histogram\0_Gambar_Referensi\Gambar_Referensi.BMP"
reference_image = np.array(Image.open(ref_img_path).convert('L'))
ref_hist = generate_histogram(reference_image)

def process_images(input_folder, output_folder, ref_hist):
    """
    Processes all images in the input folder by matching their histograms
    to the reference histogram and saves the results to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    input_images = [
        img for img in os.listdir(input_folder)
        if img.lower().endswith(('.png', '.jpg', '.bmp'))
    ]
    
    for img_name in input_images:
        input_img_path = os.path.join(input_folder, img_name)
        input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

        # Perform histogram matching
        img_matched = match_histogram(input_img, ref_hist)
        img_matched = np.uint8(img_matched)  

        output_file_name = f"histogram_databaru_{img_name}"
        output_path = os.path.join(output_folder, output_file_name)
        Image.fromarray(img_matched).save(output_path)

    print(f"Processed images saved in: {output_folder}")

# Process NG and OK folders
process_images(ng_folder, ng_output_folder, ref_hist)
process_images(ok_folder, ok_output_folder, ref_hist)
