import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# TODO: maybe make into a class ...

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

# TODO: rewrite to put dist[0] elsewhere
def find_distances(coord, dist):
    dist[0] = abs(coord[0] - pacman_coord[0]) + abs(coord[1] - pacman_coord[1])
    # print("dist", dist)

def find_blue_ghosts(img):
    image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_blue = np.array([60, 100, 188])
    upper_blue = np.array([150,255, 255])
    mask = cv.inRange(image, lower_blue, upper_blue)
    # print("nonzero ", np.count_nonzero(mask))
    return np.count_nonzero(mask) > 0

def check_pills():
    for i in range(4):
        if(abs(pacman_coord[0] - pill_locs[i][0]) <= 3 and abs(pacman_coord[1] - pill_locs[i][1]) <= 3):
            pill_eaten[i] = True
        pill_dist[i] = abs(pacman_coord[0] - pill_locs[i][0]) + abs(pacman_coord[1] - pill_locs[i][1]) 

# Declare colours. OpenCV uses BGR not RGB
pacman_colour = [74, 164, 210]
pink_ghost_colour = [179, 89, 198]
red_ghost_colour = [72, 72, 200]
# called blue ghost sometimes?
green_ghost_colour = [153, 184, 84]
orange_ghost_colour = [48, 122, 180]
# 116 pixels per ghost, estimate as circle means radius is about 5 pixels
dark_blue_ghost = [194, 114, 66]


# Declare and initialize coordinates 
pacman_coord = [0, 0]
pink_ghost_coord = [0, 0]
red_ghost_coord = [0, 0]
green_ghost_coord = [0, 0]
orange_ghost_coord = [0, 0]


# Declare distances 
# TODO: make into one array :(
to_pink_ghost = [0]
to_red_ghost = [0]
to_green_ghost = [0]
to_orange_ghost = [0]

# Declare pill info
power_pill_top_left = [19.5, 18]
power_pill_btm_left = [19.5, 150]
power_pill_top_right = [300.5, 18]
power_pill_btm_right = [300.5, 150]
pill_locs = []
pill_locs.append(power_pill_top_left)
pill_locs.append(power_pill_top_right)
pill_locs.append(power_pill_btm_right)
pill_locs.append(power_pill_btm_left)
# pill 1,2,3,4
pill_eaten = [False, False, False, False]
# top left, top right, btm right, btm left
pill_dist = [0,0,0,0]


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

    check_pills()

    hasBlueGhost = find_blue_ghosts(img)

    return pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost[0], to_red_ghost[0], to_green_ghost[0], to_orange_ghost[0], pill_eaten, pill_dist, hasBlueGhost

# print("pacman coord ", pacman_coord)
# print("pink ghost ", pink_ghost_coord)
# print("red ghost ", red_ghost_coord)
# print("green ghost ", green_ghost_coord)
# print("orange ghost ", orange_ghost_coord)



