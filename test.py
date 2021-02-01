# from ghost_tracker import GhostTracker
# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image
# import numpy as np
# import pandas as pd

# img = cv2.imread("blueghostfour.png")
# im = plt.imread('blueghostfour.png')

# imframe = Image.open('blueghostfour.png')
# npframe = np.array(imframe.getdata())
# imgrgbdf = pd.DataFrame(npframe)

# # imagePeeler = GhostTracker()
# # print("About to seek pacman")
# # characters, bg_locs = imagePeeler.wheresPacman(imgrgbdf)
# # [194, 114, 66]
# lower_blue = np.array([190, 110, 60])
# upper_blue = np.array([200, 120, 70])
# lower_red = np.array([160,20,70])
# upper_red = np.array([190,255,255])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(img, lower_blue, upper_blue)
# result = img.copy()
# result = cv2.bitwise_and(result, result, mask=mask)

# cv2.imshow('ogimg', img)
# cv2.imshow('mask', mask)
# cv2.imshow('result', result)
# cv2.waitKey()

import numpy as np
import cv2

image = cv2.imread('blueghostfour.png')
result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([155,25,0])
upper = np.array([179,255,255])
# BGR # [194, 114, 66] [60, 100, 188], [70, 120, 200]
# 218 66 76
lower_blue = np.array([60, 100, 188])
upper_blue = np.array([255,255, 255])
mask = cv2.inRange(image, lower_blue, upper_blue)
# mask = cv2.inRange(image, np.array([100, 0, 0]), np.array([100, 0, 0]))
result = cv2.bitwise_and(result, result, mask=mask)
print("nonzero ", np.count_nonzero(mask))
with open("mask.txt", "w") as text_file:
        text_file.write(str(mask))
cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()