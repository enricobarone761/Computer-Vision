import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imR = cv.imread('/Users/enricobarone/Desktop/Computer-Vision/Progetto Esame/Assignment_1/DATASET/val/c1/zh3_01_01_R.png')
imT = cv.imread('/Users/enricobarone/Desktop/Computer-Vision/Progetto Esame/Assignment_1/DATASET/val/c1/zh3_01_01_T.png')

# Convert images to grayscale
imR_gray = cv.cvtColor(imR, cv.COLOR_BGR2GRAY)
imT_gray = cv.cvtColor(imT, cv.COLOR_BGR2GRAY)

# Flatten the images to 1D arrays
imR_flat = imR_gray.flatten()
imT_flat = imT_gray.flatten()


def entropia(img_flat, bins=128):
    hist_s, _ = np.histogram(img_flat, bins, density=True)
    return -np.sum(hist_s[hist_s > 0] * np.log2(hist_s[hist_s > 0]))


def entropia_congiunta(img1_flat, img2_flat, bins=128):
    hist, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins, density=True)
    return -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))


mutua_informazione = entropia(imR_flat) + entropia(imT_flat) - entropia_congiunta(imR_flat, imT_flat)
print(mutua_informazione)

# Plot the histogram
hist_2d, _, _ = np.histogram2d(imR_flat, imT_flat, bins=128)
plt.imshow(hist_2d, origin='lower', cmap='hot')
plt.show()