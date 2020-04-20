# Utility Functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List

def add_noise(img: np.array, min_noise_factor: float=.3, 
              max_noise_factor: float=.6) -> np.array:
    """Add random noise to an image using a uniform distribution
    Input
    -----
        img: array-like
            Array of pixels
        min_noise+factor: float
            Min value for random noise
        max_noise_factor: float
            Max value for random noise
    Output
    ------
        Noisy image"""
    # Generating and applying noise to image:
    noise_factor = np.random.uniform(min_noise_factor, max_noise_factor)
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape) 
    img_noisy = img + noise
    
    # Making sure the image value are still in the proper range:
    img_noisy = np.clip(img_noisy, 0., 1.)
    
    return img_noisy

def plot_image_grid(images: List[np.array], titles:List[str]=None, 
                    figure: plt.figure =None,grayscale: bool = False, 
                    transpose: bool = False) -> plt.figure:
    """
    Plot a grid of n x m images
    Input
    -----
        images: List[np.array]  
            Images in a n x m array
        titles: List[str] (opt.) 
            List of m titles for each image column
        figure: plt.figure (opt.) 
            Pyplot figure (if None, will be created)
        grayscale: bool (opt.) 
            If True return grayscaled images
        transpose: bool (opt.) 
            If True, transpose the grid
    Output
    ------
        Pyplot figure filled with the images
    """
    num_cols, num_rows = len(images), len(images[0])
    img_ratio = images[0][0].shape[1] / images[0][0].shape[0]

    if transpose:
        vert_grid_shape, hori_grid_shape = (1, num_rows), (num_cols, 1)
        figsize = (int(num_rows * 5 * img_ratio), num_cols * 5)
        wspace, hspace = 0.2, 0.
    else:
        vert_grid_shape, hori_grid_shape = (num_rows, 1), (1, num_cols)
        figsize = (int(num_cols * 5 * img_ratio), num_rows * 5)
        hspace, wspace = 0.2, 0.
    
    if figure is None:
        figure = plt.figure(figsize=figsize)
    imshow_params = {'cmap': plt.get_cmap('gray')} if grayscale else {}
    grid_spec = gridspec.GridSpec(*hori_grid_shape, wspace=0, hspace=0)
    for j in range(num_cols):
        grid_spec_j = gridspec.GridSpecFromSubplotSpec(
            *vert_grid_shape, subplot_spec=grid_spec[j], wspace=wspace, hspace=hspace)

        for i in range(num_rows):
            ax_img = figure.add_subplot(grid_spec_j[i])
            # ax_img.axis('off')
            ax_img.set_yticks([])
            ax_img.set_xticks([])
            if titles is not None:
                if transpose:
                    ax_img.set_ylabel(titles[j], fontsize=25)
                else:
                    ax_img.set_title(titles[j], fontsize=15)
            ax_img.imshow(images[j][i], **imshow_params)

    figure.tight_layout()
    return figure
