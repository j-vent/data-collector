import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('data_2021-01-27_10-50-58\screen\screenshot1.png',0)

# change dim??
edges = cv.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('OG img'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge img'), plt.xticks([]), plt.yticks([])
plt.show()