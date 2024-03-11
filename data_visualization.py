from tools_lib import *

# plot 5x5 grid of images
def plot_matrix_grid(V):
    """
    Given an array V containing stacked matrices, plots them in a grid layout.
    V should have shape (K,M,N) where V[k] is a matrix of shape (M,N).
    """
    assert V.ndim == 3, "Expected V to have 3 dimensions, not %d" % V.ndim
    k, m, n = V.shape
    ncol = 5                                     # At most 5 columns
    nrow = min(5, (k + ncol - 1) // ncol)        # At most 5 rows
    V = V[:nrow*ncol]                            # Focus on just the matrices we'll actually plot
    figsize = (2*ncol, max(1, 2*nrow*(m/n)))     # Guess a good figure shape based on ncol, nrow
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=figsize)
    vmin, vmax = np.percentile(V, [0.1, 99.9])   # Show the main range of values, between 0.1%-99.9%
    for v, ax in zip(V, axes.flat):
        img = ax.matshow(v, vmin=vmin, vmax=vmax, cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(img, cax=fig.add_axes([0.92, 0.25, 0.01, .5]))   # Add a colorbar on the right

    plt.show()

def plot_image_w_pixel_density(image, label):
    assert image.ndim  == 2, "Expected image to have 2 dimensions, not %d" % image.ndim
    assert image.shape == (48, 48), "Expected image to be of size 48x48, not %d" % image.ndim

    plt.figure(figsize=(10, 5))

    # image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.axis('off')

    # pixel density
    plt.subplot(1, 2, 2)
    plt.hist(image.reshape(48*48), bins=256, color='gray')
    plt.title('Pixel Density Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_unique_emotions_densities(X, y, unique_emotions):
    plt.figure(figsize=(14, 7))

    for i in range(len(unique_emotions)):
        locs = (np.where(y == unique_emotions[i])[0]) # indices where y is = some emotion (a tuple is returned, then pick the first of the tuple)
        loc  = locs[0]

        plt.subplot(2, len(unique_emotions), i+1)
        plt.imshow(X[loc].reshape(48, 48), cmap='gray')
        plt.title(y[loc])
        plt.axis('off')

    for i in range(len(unique_emotions)):
        locs = (np.where(y == unique_emotions[i])[0]) # indices where y is = some emotion (a tuple is returned, then pick the first of the tuple)
        loc  = locs[0]

        plt.subplot(2, len(unique_emotions), len(unique_emotions)+i+1)
        plt.hist(X[loc], bins=256, color='gray')
        plt.title('Pixel Density Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True)

    plt.tight_layout()
    plt.show()