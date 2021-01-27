import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# extra param for edge detection
#img = cv.imread('data_2021-01-27_10-50-58\screen\screenshot112.png',0)

# image for colour detection
# 2 ghosts
# im = cv.imread('data_2021-01-27_10-50-58\screen\screenshot1.png')
#im = cv.imread('data_2021-01-27_10-50-58\screen\screenshot112.png')

# def find_element_centroid(colour, coord):
#     y,x = np.where(np.all(im == colour, axis=2))
#     pairs = []
#     for i in range(len(x)):
#         pairs.append([x[i],y[i]])
#     # print("x ", x, " y ", y)
#     coordx, coordy = np.mean(pairs, axis = 0)
#     coord[0] = coordx
#     coord[1] = coordy
#     # print("centroid ", coordx, ", ", coordy)

def find_element_centroid(img, colour, coord):
    y,x = np.where(np.all(img == colour, axis=2))
    pairs = []
    for i in range(len(x)):
        pairs.append([x[i],y[i]])
    # print("x ", x, " y ", y)
    if(len(x) != 0 and len(y) != 0):
        coordx, coordy = np.mean(pairs, axis = 0)
        coord[0] = round(coordx)
        coord[1] = round(coordy)
    # print("centroid ", coordx, ", ", coordy)

# Declare colours 
# OpenCV uses BGR
pacman_colour = [74, 164, 210]
pink_ghost_colour = [179, 89, 198]
red_ghost_colour = [72, 72, 200]
# called blue ghost sometimes?
green_ghost_colour = [153, 184, 84]
orange_ghost_colour = [48, 122, 180]

# Declare and initialize coordinates 
pacman_coord = [0, 0]
pink_ghost_coord = [0, 0]
red_ghost_coord = [0, 0]
green_ghost_coord = [0, 0]
orange_ghost_coord = [0, 0]

def find_all_coords(im):
    img = cv.imread(im)
    # print("img ", img)
    find_element_centroid(img, pacman_colour, pacman_coord)
    find_element_centroid(img, pink_ghost_colour, pink_ghost_coord)
    find_element_centroid(img, red_ghost_colour, red_ghost_coord)
    find_element_centroid(img, green_ghost_colour, green_ghost_coord)
    find_element_centroid(img, orange_ghost_colour, orange_ghost_coord)
    return pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord

# print("pacman coord ", pacman_coord)
# print("pink ghost ", pink_ghost_coord)
# print("red ghost ", red_ghost_coord)
# print("green ghost ", green_ghost_coord)
# print("orange ghost ", orange_ghost_coord)

# plot img and edge detection
# edges = cv.Canny(img,100,200)

# plt.figure()
# # flip because opencv is BGR
# plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB)) 
# plt.title('OG img'), plt.xticks([]), plt.yticks([])
# plt.show()

# 210, 164, 74
