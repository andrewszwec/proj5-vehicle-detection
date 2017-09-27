import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    # If colour image then turn to greyscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                    pixels_per_cell=(pix_per_cell, pix_per_cell), 
                                    cells_per_block=(cell_per_block, cell_per_block), 
                                    visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:
        features, _ = hog(img, orientations=orient, 
                                    pixels_per_cell=(pix_per_cell, pix_per_cell), 
                                    cells_per_block=(cell_per_block, cell_per_block), 
                                    visualise=False, feature_vector=feature_vec)
        return features



# Read a color image
if __name__ == '__main__':
    img = cv2.imread("../test_images/25.png")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    features, img = get_hog_features(img, vis=True, feature_vec=True)
    plt.imshow(img, cmap='gray')
    plt.show()

