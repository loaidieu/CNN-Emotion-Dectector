from data_processing import *
from data_visualization import *

##################################################################################
#                               Data Processing                                  #
##################################################################################
# add more data by flipping the images (data augmentation)
# can only be done once
if (check_flipped('./data') == False): flip_png('./data')

# convert images into numpy arrays
X, y = png_to_numpy('./data')

##################################################################################
#                            Data Visualization                                  #
##################################################################################
# get all the unique emotions and their counts
unique_emotions, counts = np.unique(y, return_counts=True)

# plot the distribution of labels
plt.figure(figsize=(6,4))
plt.bar(unique_emotions, counts)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.xticks(unique_emotions)  # ensure all labels are displayed on the x-axis
plt.show()

# visualize a sample image of each category along with its pixel density distribution
plot_unique_emotions_densities(X, y, unique_emotions)

# plot 25 images (focused)
locs = (np.where(y == 'focused')[0])
plot_matrix_grid(X.reshape(-1, 48, 48)[locs])
plt.show()

# plot 25 images (happy)
locs = (np.where(y == 'happy')[0])
plot_matrix_grid(X.reshape(-1, 48, 48)[locs])

# plot 25 images (neutral)
locs = (np.where(y == 'neutral')[0])
plot_matrix_grid(X.reshape(-1, 48, 48)[locs])

# plot 25 images (surprised)
locs = (np.where(y == "surprised")[0])
plot_matrix_grid(X.reshape(-1, 48, 48)[locs])

##################################################################################
#                            Data Normalization                                  #
##################################################################################
# normalize the features
scaler = sklearn.preprocessing.StandardScaler()

# fit the dataset to the scaler
scaler.fit(X)

# scale X and replace it with its original counterpart
X = scaler.transform(X)

# verify that the scaling was successful
plot_unique_emotions_densities(X, y, unique_emotions)