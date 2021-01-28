import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def find_element_centroid(img, colour, coord):
    y,x = np.where(np.all(img == colour, axis=2))
    pairs = []
    for i in range(len(x)):
        pairs.append([x[i],y[i]])
   
    if(len(x) != 0 and len(y) != 0):
        # calculate centroid
        coordx, coordy = np.mean(pairs, axis = 0)
        coord[0] = round(coordx)
        coord[1] = round(coordy)

def find_distances(coord, dist):
    dist[0] = abs(coord[0] - pacman_coord[0]) + abs(coord[1] - pacman_coord[1])
    print("dist", dist)

# def check_pills():
#     for()
#     if(pacman_coord[0])

# Declare colours. OpenCV uses BGR not RGB
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

power_pill_top_left = [19.5, 18]
power_pill_btm_left = [19.5, 150]
power_pill_top_right = [300.5, 18]
power_pill_btm_right = [300.5, 150]
pill_locs = []
pill_locs.append(power_pill_top_left)
pill_locs.append(power_pill_top_right)

# Declare distances 
to_pink_ghost = [0]
to_red_ghost = [0]
to_green_ghost = [0]
to_orange_ghost = [0]
to_pill_one = 0 # top left
to_pill_two = 0 # top right
to_pill_three = 0 # btm right
to_pill_four = 0 # btm left

pill_one_eaten = False
pill_two_eaten = False
pill_three_eaten = False
pill_four_eaten = False

def find_all_coords(im):
    img = cv.imread(im)
    # img_plot = cv.imread(im,0)
    # # plot img and edge detection
    # edges = cv.Canny(img_plot,100,200)
    # plt.figure()
    # # flip because opencv is BGR
    # plt.imshow(cv.cvtColor(img_plot, cv.COLOR_BGR2RGB)) 
    # plt.title('OG img'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # move to own func
    find_element_centroid(img, pacman_colour, pacman_coord)
    find_element_centroid(img, pink_ghost_colour, pink_ghost_coord)
    find_distances(pink_ghost_coord, to_pink_ghost)
    find_element_centroid(img, red_ghost_colour, red_ghost_coord)
    find_distances(red_ghost_coord, to_red_ghost)
    find_element_centroid(img, green_ghost_colour, green_ghost_coord)
    find_distances(green_ghost_coord, to_green_ghost)
    find_element_centroid(img, orange_ghost_colour, orange_ghost_coord)
    find_distances(orange_ghost_coord, to_orange_ghost)

    # check_pills()

    return pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost[0], to_red_ghost[0], to_green_ghost[0], to_orange_ghost[0]

# print("pacman coord ", pacman_coord)
# print("pink ghost ", pink_ghost_coord)
# print("red ghost ", red_ghost_coord)
# print("green ghost ", green_ghost_coord)
# print("orange ghost ", orange_ghost_coord)



