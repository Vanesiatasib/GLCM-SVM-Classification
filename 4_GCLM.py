import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Input folders
ng_folder = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\3_Histogram\NG"
ok_folder = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\3_Histogram\OK"

# GLCM configuration
DISTANCES = [15]
ANGLES = np.deg2rad([0])
FEATURE_NAMES = ['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', 'dissimilarity']

def glcm_features(image, distances=DISTANCES, angles=ANGLES):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = [graycoprops(glcm, prop)[0, 0] for prop in FEATURE_NAMES]
    return features

def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.BMP'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Apply Gaussian blur with a 3x3 kernel
                img = cv2.GaussianBlur(img, (3, 3), 0)
                images.append(img)
                filenames.append(filename)
            else:
                print(f"Failed to load image: {img_path}")
    return images, filenames


def extract_glcm_features_from_images(images):
    features = []
    for i, img in enumerate(images):
        try:
            features.append(glcm_features(img))
        except Exception as e:
            print(f"Error processing image at index {i}: {e}")
    return np.array(features)

def save_features_to_csv(features, filenames, label, output_path):
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    df['filename'] = filenames
    df['label'] = label
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    return df

def main():
    # Load and process NG images
    ng_images, ng_filenames = load_images_from_folder(ng_folder)
    ng_features = extract_glcm_features_from_images(ng_images)
    ng_csv_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\4_GLCM\ng_glcm_features.csv"
    df_ng = save_features_to_csv(ng_features, ng_filenames, 'NG', ng_csv_path)

    # Load and process OK images
    ok_images, ok_filenames = load_images_from_folder(ok_folder)
    ok_features = extract_glcm_features_from_images(ok_images)
    ok_csv_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\4_GLCM\ok_glcm_features.csv"
    df_ok = save_features_to_csv(ok_features, ok_filenames, 'OK', ok_csv_path)

    # Combine and save NG and OK features
    combined_csv_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\03_Data_Baru\4_GLCM\Histogram_databaru_d_15_t_0.csv"
    df_combined = pd.concat([df_ng, df_ok], ignore_index=True)
    df_combined.to_csv(combined_csv_path, index=False)
    print(f"Combined GLCM features saved to {combined_csv_path}")

if __name__ == "__main__":
    main()
