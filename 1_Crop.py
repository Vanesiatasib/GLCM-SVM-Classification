import os
import cv2

def crop_and_save_images(input_folder, output_folder, crop_coords, label):
    x, y, w, h = crop_coords
    os.makedirs(output_folder, exist_ok=True)
    image_list = os.listdir(input_folder)

    for image_name in image_list:
        img_path = os.path.join(input_folder, image_name)
        img = cv2.imread(img_path)
        
        if img is not None: 
            roi_crop = img[y:y+h, x:x+w]
            output_path = os.path.join(output_folder, f"crop_{label}_{image_name}")
            cv2.imwrite(output_path, roi_crop)
            print("Hasil crop disimpan di:", output_path)
        else:
            print(f"Could not load image: {img_path}")

ng_folder = r"D:\Proposal Skripsi gas 2024\skripsi\data_baru\0_Data_Awal\ng"
ok_folder = r"D:\Proposal Skripsi gas 2024\skripsi\data_baru\0_Data_Awal\ok"
# Define output folders for NG and OK categories
ng_output_folder = r"D:\Proposal Skripsi gas 2024\skripsi\data_baru\1_Data_Crop\NG"

ok_output_folder = r"D:\Proposal Skripsi gas 2024\skripsi\data_baru\1_Data_Crop\OK"
# ng_crop_coords = (576, 446, 95, 174)
# ok_crop_coords = (590, 456, 100, 170)

ng_crop_coords = (576, 446, 95, 174)
ok_crop_coords = (590, 456, 95, 174)

crop_and_save_images(ng_folder, ng_output_folder, ng_crop_coords, 'NG')
crop_and_save_images(ok_folder, ok_output_folder, ok_crop_coords, 'OK')
