# board_detection.py
# author : joseph Greshik
# this file should be able to take in an input image and 
#   find lines for all chess board squares
#   apply k-means clustering to lines to fit some expected board structure
#   draw lines on original image
#   return a board object relative to image scene (definition of object needed)

import cv2, sys, math, os
import numpy as np
from sklearn.cluster import KMeans

# Display the image in the given named window.  Allow for resizing, and
# assign the mouse callback function so that it prints coordinates on a click.

WIN_MAX_SIZE = 2048 / 2

def print_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y, " : ", param[y, x])

def display_image(win_name, image):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # Create window; allow resizing
    h = image.shape[0]  # image height
    w = image.shape[1]  # image width

    # Shrink the window if it is too big (exceeds some maximum size).
    if max(w, h) > WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE / max(w, h)
    else:
        scale = 1
    cv2.resizeWindow(winname=win_name, width=int(w * scale), height=int(h * scale))

    # Assign callback function, and show image.
    cv2.setMouseCallback(window_name=win_name, on_mouse=print_xy, param=image)
    cv2.imshow(win_name, image)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("WORKING IN : "+dir_path)

data_path=dir_path+'/../data/'
print("data path : "+data_path)

# detect_lines()
#   apply input_image -> gaussian blur -> canny edge detection -> detect hough lines and populate houghLines param with image lines in hough space
# <args> are 
# input binary image
# verbosity (show images with detected lines)
# Constants.
#   hough threshold
#   lower canny threshold
#   guassian kernel size
# <return> is 
#   input image 
def detect_lines(bgr_image,HOUGHTHRESH=100,LOW_THRESH_CANNY=100.0,KSIZE=(9,9)):

    gray_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2GRAY)

    # Smooth the image with a Gaussian filter.
    # kernel size is given above constant KSIZE
    gray_image=cv2.GaussianBlur(
        src=gray_image,
        ksize=KSIZE, # kernel size (should be odd numbers; if 0, compute it from sigma)
        sigmaX=0)  

    # Run edge detection.
    edges_image = cv2.Canny(
        image=gray_image,
        apertureSize=3,  # size of Sobel operator
        threshold1=LOW_THRESH_CANNY,  # lower threshold
        threshold2=3 * LOW_THRESH_CANNY,  # upper threshold
        L2gradient=True)  # use more accurate L2 norm

    # Run Hough transform.  The output houghLines has size (N,1,2), where N is #lines.
    # The 3rd dimension has values rho,theta for the line.
    houghLines = cv2.HoughLines(
        image=edges_image,
        rho=1,  # Distance resolution of the accumulator in pixels
        theta=math.pi / 180,  # Angle resolution of the accumulator in radians
        threshold=HOUGHTHRESH  # Accumulator threshold (get lines where votes>threshold)
    )

    if houghLines is not None:
        
        # draw all lines
        for i in range(len(houghLines)): 
            rho = houghLines[i][0][0]  # distance from (0,0)
            theta = houghLines[i][0][1]  # angle in radians
            color=(0, 0, 255)
            draw_line(rho,theta,bgr_image,color)

        return houghLines,bgr_image
    else:
        print('Error in finding hough lines')
        return None

# draw_line()
#   draw a line on an image given the image, line rho and theta values and a color for the line
def draw_line (rho, theta, image, color):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho        # Point on line, where rho vector intersects
    y0 = b * rho
    # Find two points on the line, very far away.
    p1 = (int(x0 + 4000 * (-b)), int(y0 + 4000 * (a)))
    p2 = (int(x0 - 4000 * (-b)), int(y0 - 4000 * (a)))
    cv2.line(img=image, pt1=p1, pt2=p2, color=color, thickness=1, lineType=8)
    #display_image(image+" lines_image", bgr_image)
    #cv2.waitKey(0)

# scale_h_space()
#   scale input Hough space from max(abs(-rho,+rho))xpi to 180x180 like Hoff method
# <args> are
# houghLines: lines populating hough space
# <return> is 
# scaled_lines: original lines scaled rho and theta values to (float) 180x180
def scale_h_space(houghLines):
    # get absolute max value for rho in space
    max_rho=0
    for i in range(len(houghLines)): 
        rho = houghLines[i][0][0]  # distance from (0,0)
        if math.fabs(rho) > max_rho:
            max_rho=math.fabs(rho)
    # scale space rho and theta values
    for i in range(len(houghLines)): 
        # scale rho values
        houghLines[i][0][0]=houghLines[i][0][0]/max_rho*90.0+90.0
        # scale theta values
        houghLines[i][0][1]=houghLines[i][0][1]/math.pi*180.0
    # houghLines is now scaled to 180x180 space 
    # both rho and theta are within range (0,180]
    return houghLines

def cluster_lines (lines):


def main():
    # we run the program for all of the provided test images  
    images=[]
    for i in range(2):
        images.append('image0'+str(i)+'.jpg')
    for image in images:
        file_name, file_extension = os.path.splitext(image)
        print("Image : "+data_path+image)
        bgr_image = cv2.imread(data_path+image)

        lines,bgr_image=detect_lines(bgr_image)

        scale_h_space(lines)
#        for i in range(10):
#            rho = lines[i][0][0]  # distance from (0,0)
#            theta = lines[i][0][1]  # angle in radians
#            print('rho   : '+str(rho))
#            print('theta : '+str(theta))

        display_image(image+"_lines_image", bgr_image)
        cv2.waitKey(0)

main()

