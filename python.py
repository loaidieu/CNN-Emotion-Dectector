import os
from opencv import cv2

# Path to the dataset folder
dataset_path = './dataset/'

# Desired size for resizing
target_size = (48, 48)

# Iterate through subfolders
for emotion in ['surprised', 'happy', 'neutral', 'engaged']:
    subfolder_path = os.path.join(dataset_path, emotion)

    # Create a new folder to store resized images
    resized_folder_path = os.path.join(dataset_path, f'{emotion}_resized')
    os.makedirs(resized_folder_path, exist_ok=True)

    # Iterate through images in the subfolder
    for filename in os.listdir(subfolder_path):
        img_path = os.path.join(subfolder_path, filename)

        # Read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image
        resized_img = cv2.resize(img, target_size)

        # Save the resized image to the new folder
        new_img_path = os.path.join(resized_folder_path, filename)
        cv2.imwrite(new_img_path, resized_img)

print("Resizing complete!")
