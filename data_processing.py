from tools_lib import *

# images to numpy arrays conversion
def png_to_numpy(images_folder):
    # create 2 empty lists, one for features and one for labels
    features = []
    labels   = []

    for emotion_folder in sorted(os.listdir(images_folder)):
        # create path for each emotion folder
        emotion_folder_path = os.path.join(images_folder, emotion_folder)

        for filename in sorted(os.listdir(emotion_folder_path)):

            if filename.endswith('.png'):
                # open the image
                image_path = os.path.join(emotion_folder_path, filename)
                img        = Image.open(image_path)

                # convert the image into a numpy array
                img_array = np.array(img)

                # flatten the array to size 2304 (48x48)
                img_array_flat = img_array.flatten()

                # append the flattened array to the features list
                features.append(img_array_flat)

                # append the label to the labels list
                labels.append(emotion_folder)

        # lastly convert both lists into numpy arrays
        features_np = np.array(features)
        labels_np   = np.array(labels)

    return features_np, labels_np

# flipping every image in a folder for data augmentation
def flip_png(images_folder):
    for emotion_folder in sorted(os.listdir(images_folder)):
        # create path for each emotion folder
        emotion_folder_path = os.path.join(images_folder, emotion_folder)

        for filename in sorted(os.listdir(emotion_folder_path)):

            if filename.endswith('.png'):

                image_path = os.path.join(emotion_folder_path, filename)
                img        = Image.open(image_path)

                flipped_image = img.transpose(Image.FLIP_TOP_BOTTOM) # flip

                # extract the filename (without extension) from the original image path
                filename_wo_extension = os.path.splitext(image_path)[0]

                # save it
                flipped_image.save(filename_wo_extension + '_flipped.png')


def rotate_png(images_folder, rotation_angle=30):
    for emotion_folder in sorted(os.listdir(images_folder)):
        # create path for each emotion folder
        emotion_folder_path = os.path.join(images_folder, emotion_folder)

        for filename in sorted(os.listdir(emotion_folder_path)):

            if filename.endswith('.png'):

                image_path = os.path.join(emotion_folder_path, filename)
                img        = Image.open(image_path)

                # define the rotation transformation
                rotation_transform = transforms.RandomRotation(degrees=(-rotation_angle, rotation_angle))

                # apply rotation to the image
                rotated_image = rotation_transform(img)

                # extract the filename (without extension) from the original image path
                filename_wo_extension = os.path.splitext(image_path)[0]

                # save it
                rotated_image.save(filename_wo_extension + '_rotated.png')

def adjust_light_png(images_folder):
    for emotion_folder in sorted(os.listdir(images_folder)):
        # create path for each emotion folder
        emotion_folder_path = os.path.join(images_folder, emotion_folder)

        for filename in sorted(os.listdir(emotion_folder_path)):

            if filename.endswith('.png'):

                image_path = os.path.join(emotion_folder_path, filename)
                img        = Image.open(image_path)

                light_adjusted_image = transforms.functional.adjust_brightness(img, 0.5)

                # extract the filename (without extension) from the original image path
                filename_wo_extension = os.path.splitext(image_path)[0]

                # save it
                light_adjusted_image.save(filename_wo_extension + '_light_adjusted.png')

# check and stop if there are already modified images in the folder (post data augmentation)
def check_modified(images_folder):
    for emotion_folder in sorted(os.listdir(images_folder)):
        # create path for each emotion folder
        emotion_folder_path = os.path.join(images_folder, emotion_folder)

        for filename in sorted(os.listdir(emotion_folder_path)):
            if '_flipped' in filename or '_rotated' in filename or '_light_adjusted' in filename:
                return True

        return False