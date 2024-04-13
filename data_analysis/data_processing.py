from utils.tools_lib import *

# images to dictionary conversion
def png_to_dict(images_folder):
    # create 2 empty lists, one for features and one for labels
    images = {}

    for emotion_folder in sorted(os.listdir(images_folder)):
        # create path for each emotion folder
        emotion_folder_path = os.path.join(images_folder, emotion_folder)

        # skip the .DS_Store file
        if emotion_folder == '.DS_Store': continue

        # loop through all the files in the folder and find metadata csv file
        for filename in sorted(os.listdir(emotion_folder_path)):

            # metadata 
            if filename.endswith('.csv'): 
                # read the csv file
                metadata_df = pd.read_csv(os.path.join(emotion_folder_path, filename))

                if 'age' not in metadata_df.columns and 'gender' not in metadata_df.columns:
                    # convert the file_attributes column to a dictionary
                    metadata_df['file_attributes'] = metadata_df['file_attributes'].apply(json.loads)

                    # make a new column for age
                    metadata_df['age'] = metadata_df['file_attributes'].apply(lambda x: x['age'])

                    # make a new column for gender
                    metadata_df['gender'] = metadata_df['file_attributes'].apply(lambda x: x['gender'])

                for index, row in metadata_df.iterrows():
                    # get the filename
                    image_filename = row['filename']

                    # open the image
                    image_path = os.path.join(emotion_folder_path, image_filename)
                    img        = Image.open(image_path)

                    # convert the image into a numpy array
                    img_array = np.array(img)

                    # flatten the array to size 2304 (48x48)
                    img_array_flat = np.array(img_array.flatten())

                    # get gender
                    img_gender = metadata_df[metadata_df['filename'] == image_filename]['gender']

                    # get age
                    img_age = metadata_df[metadata_df['filename'] == image_filename]['age']

                    # add to dictionary (key: filename, value: dictionary with np array, path, age, gender, emotion)
                    images[f'{emotion_folder}/{image_filename}'] = {'np_array': img_array_flat,
                                                                    'path': image_path,
                                                                    'age': img_age,
                                                                    'gender': img_gender,
                                                                    'emotion': emotion_folder}

    return images

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