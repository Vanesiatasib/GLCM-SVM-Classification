import os
import cv2

def convert_and_save_images(input_folder, output_folder, label):

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of image files in the input directory
    image_list = os.listdir(input_folder)

    for image_name in image_list:
        img_path = os.path.join(input_folder, image_name)
        img = cv2.imread(img_path)

        if img is not None:
            try:
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Construct the output path and save the grayscale image
                output_path = os.path.join(output_folder, f"gray_{label}_{image_name}")
                cv2.imwrite(output_path, gray_image)

                print(f"Grayscale image saved at: {output_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        else:
            print(f"Could not load image: {img_path}")

# Input folders for NG and OK categories
ng_folder = r"D:\Proposal Skripsi gas 2024\skripsi\data_baru\1_Data_Crop\NG"
ok_folder = r"D:\Proposal Skripsi gas 2024\skripsi\data_baru\1_Data_Crop\OK"

# Output folders for NG and OK categories
ng_output_folder = r"D:\Proposal Skripsi gas 2024\skripsi\data_baru\2_GrayScale\NG"
ok_output_folder = r"D:\Proposal Skripsi gas 2024\skripsi\data_baru\2_GrayScale\OK"

# Perform grayscale conversion and saving for both NG and OK categories
convert_and_save_images(ng_folder, ng_output_folder, 'NG')
convert_and_save_images(ok_folder, ok_output_folder, 'OK')
