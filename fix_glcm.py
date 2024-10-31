import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D 

# ROADMAP: input -> label -> crop -> grayscale -> gaussian -> glcm

ng_folder = r"D:\Proposal Skripsi gas 2024\skripsi\bahan_penelitian\ng"
ok_folder = r"D:\Proposal Skripsi gas 2024\skripsi\bahan_penelitian\ok"
ng_roi = {"x": 576, "y": 446, "w": 95, "h": 174}
ok_roi = {"x": 590, "y": 456, "w": 95, "h": 174}

def extract_roi(folder, roi):
    cropped_images = [] 
    for filename in os.listdir(folder):
        if filename.endswith('.BMP'): 
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                roi_crop = img[y:y + h, x:x + w]
                cropped_images.append(roi_crop)
            else:
                print(f"Error loading image {filename}")
    return cropped_images

def glcm_features(image, distances=None, angles=None):
    if distances is None:
        distances = [5]
    if angles is None:
        angles = np.deg2rad([0])
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    properties = ['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', 'dissimilarity']
    features = [graycoprops(glcm, prop)[0, 0] for prop in properties]
    return features

def extract_glcm_features_from_images(cropped_images):
    all_features = []
    for i, img in enumerate(cropped_images):
        if img is None:
            print(f"Image at index {i} is None, skipping.")
            continue
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print("The image is single-channel (grayscale).")
        blurred_image = cv2.GaussianBlur(img, (3, 3), 0)
        features = glcm_features(blurred_image)
        all_features.append(features)
    return np.array(all_features)


def main():
    cropped_ng_images = extract_roi(ng_folder, ng_roi)
    cropped_ok_images = extract_roi(ok_folder, ok_roi)
    
    ng_glcm_features = extract_glcm_features_from_images(cropped_ng_images)
    ok_glcm_features = extract_glcm_features_from_images(cropped_ok_images)

    feature_names = ['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'ASM', 'Dissimilarity']

    df_ng_features = pd.DataFrame(ng_glcm_features, columns=feature_names)
    df_ok_features = pd.DataFrame(ok_glcm_features, columns=feature_names)

    # ng_csv_path = r"D:\Proposal Skripsi gas 2024\skripsi\bahan_penelitian\ng_glcm_features.csv"
    # ok_csv_path = r"D:\Proposal Skripsi gas 2024\skripsi\bahan_penelitian\ok_glcm_features.csv"
    # combined_csv_path = r"D:\Proposal Skripsi gas 2024\skripsi\bahan_penelitian\name_d_5_t_0.csv"

    # df_ng_features['label'] = 'NG'
    # df_ok_features['label'] = 'OK'
    # df_ng_features['filename'] = [f for f in os.listdir(ng_folder) if f.endswith('.BMP')]
    # df_ok_features['filename'] = [f for f in os.listdir(ok_folder) if f.endswith('.BMP')]


    # df_combined = pd.concat([df_ng_features, df_ok_features], ignore_index=True)

    # df_combined.to_csv(combined_csv_path, index=False)
    # print(f"Combined GLCM features saved to {combined_csv_path}")

    # df_ng_features.to_csv(ng_csv_path, index=False)
    # df_ok_features.to_csv(ok_csv_path, index=False)
    # print(f"NG GLCM features saved to {ng_csv_path}")
    # print(f"OK GLCM features saved to {ok_csv_path}")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create 3D axes

    ax.scatter(df_ng_features['Homogeneity'], 
               df_ng_features['Energy'], 
               df_ng_features['ASM'], 
               color='blue', marker='o', label='NG Images')

    ax.scatter(df_ok_features['Homogeneity'], 
               df_ok_features['Energy'], 
               df_ok_features['ASM'], 
               color='red', marker='^', label='OK Images')

    ax.set_title('GLCM Features: 3D Scatter Plot (NG vs OK)')
    ax.set_xlabel('Homogeneity')
    ax.set_ylabel('Energy')
    ax.set_zlabel('ASM')

    ax.legend()
    plt.savefig('Cor_En_ASM.eps', format='eps')
    plt.show()

if __name__ == "__main__":
    main()
