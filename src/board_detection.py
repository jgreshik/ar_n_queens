#!/usr/bin/env python
# coding: utf-8

# In[1]:


# board_detection.py
# author : joseph Greshik
# this file should be able to take in an input image and
#   find lines for all chess board squares
#   apply k-means clustering to lines to fit some expected board structure
#   draw lines on original image
#   return a board object relative to image scene (definition of object needed)

import cv2, sys, math, os, time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from glob import glob

np.set_printoptions(precision=3)


# In[2]:


ext=".png"
# num_imgs=5
# path_name="../data/unprocessed/*"+ext
# path_name="../data/image0*"#+ext
path_name="../data/good/*"#+ext
images=glob(path_name)
save_path="../data/save/"
# images=images[0:num_imgs]


# In[3]:


max_rho=0
max_x=0
max_y=0
discon=False


# In[4]:


# constants for detecting lines
HOUGHTHRESH=55
LOW_THRESH_CANNY=20.0
KSIZE=(41,41)

similarity_threshold={
        'rho'   :   40,
        'theta' :   0.4}

d=25
sigmaColor=300
sigmaSpace=300


# In[5]:


# Display the image in the given named window.  Allow for resizing, and
# assign the mouse callback function so that it prints coordinates on a click.
WIN_MAX_SIZE = 2048 / 2


# In[6]:


def print_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y, " : ", param[y, x])


# In[7]:


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


# In[8]:


def display_plot(name,image,size=(30,30)):
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize = size)
    plt.imshow(RGB_image)
    plt.title(name)
    plt.show()


# In[9]:


def save_image (file_name,bgr_image):
    cv2.imwrite(save_path+os.path.basename(file_name)+ext,bgr_image)


# In[10]:


def get_edges_image(bgr_image,HOUGHTHRESH=HOUGHTHRESH,
                    LOW_THRESH_CANNY=LOW_THRESH_CANNY,KSIZE=KSIZE,d=d,
                    sigmaColor=sigmaColor,sigmaSpace=sigmaSpace):

    gray_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2GRAY)

    # Smooth the image with a Gaussian filter.
    # kernel size is given above constant KSIZE
    gray_image=cv2.GaussianBlur(
        src=gray_image,
        ksize=KSIZE, # kernel size (should be odd numbers; if 0, compute it from sigma)
        sigmaX=0)

#    gray_image=cv2.bilateralFilter(
#            src=gray_image,
#            d=d,
#            sigmaColor=sigmaColor,
#            sigmaSpace=sigmaSpace)

    # Run edge detection.
    edges_image = cv2.Canny(
        image=gray_image,
        apertureSize=3,  # size of Sobel operator
        threshold1=LOW_THRESH_CANNY,  # lower threshold
        threshold2=3 * LOW_THRESH_CANNY,  # upper threshold
        L2gradient=True)  # use more accurate L2 norm
    
    return edges_image


# In[11]:


def detect_lines(bgr_image,HOUGHTHRESH=HOUGHTHRESH,LOW_THRESH_CANNY=LOW_THRESH_CANNY,KSIZE=KSIZE):

    edges_image=get_edges_image(bgr_image)

    # save_image("edges"+str(time.time()),edges_image)
    
    # Run Hough transform.  The output houghLines has size (N,1,2), where N is #lines.
    # The 3rd dimension has values rho,theta for the line.
    houghLines = cv2.HoughLines(
        image=edges_image,
        rho=1,  # Distance resolution of the accumulator in pixels
        theta=math.pi / 180,  # Angle resolution of the accumulator in radians
        threshold=HOUGHTHRESH  # Accumulator threshold (get lines where votes>threshold)
    )

    if houghLines is not None:
        # remove duplicate lines
        # hough lines are stored in decreasing strength fashion
        # so we will search them from strongest -> weakest and check rho and theta difference thresholds
        # if weaker lines are found to be too similar stronger lines, their index is added to bad_inds
        bad_inds=[]
        for i in range(len(houghLines)):
            for j in range(i+1, len(houghLines)):
                if (abs(houghLines[i][0][0]-houghLines[j][0][0])<similarity_threshold['rho'] 
                    and abs(houghLines[i][0][1]-houghLines[j][0][1])<similarity_threshold['theta']):
                    bad_inds.append(j)
        # now we simply find all of the good lines using
        # a set difference with all indices in houghLines
        all_inds=[k for k in range(len(houghLines))]
        good_inds=list(set(all_inds)-set(bad_inds))
#         print(good_inds)
#         print(houghLines.shape)
#         print(houghLines[0])
        return houghLines[good_inds].squeeze()
    else:
        print('Error in finding hough lines')
        return None


# In[12]:


# draw_line()
#   draw a line on an image given the image, line rho and theta values and a color for the line
def draw_line (rho, theta, image, color, thickness=3):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho        # Point on line, where rho vector intersects
    y0 = b * rho
    # Find two points on the line, very far away.
    p1 = (int(x0 + 4000 * (-b)), int(y0 + 4000 * (a)))
    p2 = (int(x0 - 4000 * (-b)), int(y0 - 4000 * (a)))
    cv2.line(img=image, pt1=p1, pt2=p2, color=color, thickness=thickness, lineType=8)
    #display_image(image+" lines_image", bgr_image)
    #cv2.waitKey(0)


# In[13]:


def draw_lines (lines, image, color=(0, 0, 255)):
    for i in range(len(lines)):
        rho = lines[i][0]  # distance from (0,0)
        theta = lines[i][1]  # angle in radians
        draw_line(rho,theta,image,color)


# In[14]:


test=np.array([0,30,1,3,4])


# In[15]:


# we scale hough space to normalized euclidean space Q1
def hough_to_euc(houghLines,verbose=False,tag=None):
#     x_bound=0.5
#     y_shift=rho_bound
#     theta_bound=1.0
    eucLines=houghLines.copy()
    global max_x, max_y
    global discon
    max_x=0
    max_y=0
    min_x=1e9
    min_y=1e9
    # scale space rho and theta values
    # and map to euclidean space
    for i in range(len(eucLines)):
        # if rho is a negative value
        if eucLines[i][0]<0:
            # make rho a positive value
            eucLines[i][0]=math.fabs(eucLines[i][0])
            # shift theta by +pi
            eucLines[i][1]+=math.pi
        # now map to euclidean space
        # going from (rho,theta) to (x,y)
        rho=eucLines[i][0]
        theta=eucLines[i][1]
        # transform rho values
        eucLines[i][0]=math.cos(theta)*rho
        eucLines[i][1]=math.sin(theta)*rho
    # get absolute max value for x,y in space

    for i in range(len(eucLines)):
        x = eucLines[i][0] 
        y = eucLines[i][1] 
        if x > max_x:max_x=x
        elif x < min_x:min_x=x
        if y > max_y:max_y=y
        elif y < min_y:min_y=y
    # scale euclidian space by max(x,y)
    eucLines=eucLines.T
    eucLines[0]=(eucLines[0]-min_x)/(max_x-min_x)
    eucLines[1]=(eucLines[1]-min_y)/(max_y-min_y)
    eucLines=eucLines.T
    # eucLines is now scaled to new space
    # and all lines in eucLines correspond 
    # to lines in houghLines 
#     print(eucLines)

    # now check for recurring, very small x values
    # if we have many, that means we are in line with
    # the camera and should use a different distance metric
    x_check_count=0
    x_check_count_threshold=7
    small_x_value_threshold=0.05
    large_x_value_threshold=0.95
    for i in range(len(eucLines)):
        x = eucLines[i][0] 
        if x < small_x_value_threshold or x > large_x_value_threshold: x_check_count+=1
    if x_check_count>x_check_count_threshold: discon=True
    
    if verbose:
        fig = plt.figure(figsize = (10,10))
        if tag is not None:
            plt.title("Euclidean Space"+str(tag), fontsize=26)
        else: plt.title("Euclidean Space", fontsize=26)
        plt.xlabel('x', fontsize=26)
        plt.ylabel('y     ', fontsize=26,rotation=0)
        plt.scatter(eucLines[:,0],eucLines[:,1])
        if tag is not None: plt.savefig(save_path+"Hough Space"+str(tag)+ext)
        plt.show()
        
    return eucLines


# In[16]:


def cluster_lines (lines,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(lines)
    return kmeans.labels_


# In[17]:


# apply transform to hough space and remove 
# outlier lines, bias in theta dimension
def get_l1_prune_indices (in_lines, scale_factor=1.7,verbose=False):
    lines=in_lines.copy()
    max_theta=-1e9
    min_theta=1e9
    for i in range(len(lines)):
        theta = lines[i][1] 
        if math.fabs(theta) > max_theta:
            max_theta=math.fabs(theta)
        elif math.fabs(theta) < min_theta:
            min_theta=math.fabs(theta)
    lines=lines.T
    lines[1]=(lines[1]-min_theta)/(max_theta-min_theta)
    lines=lines.T
    size=len(lines)
#     print(lines)
    distances=[0 for i in range(size)]
    for i in range(size):
        for j in range(size):
            if i==j: continue
            distances[i]+=np.linalg.norm(lines[i][1]-lines[j][1])
        distances[i]/=(size-1)
#     print(distances)
    avg_dist=0
    avg_dist=np.sum(distances)/size
    bad_inds=[]
    for i in range(size):
        if distances[i] > avg_dist * scale_factor:
            bad_inds.append(i)
    bad_lines=lines[bad_inds]
    if verbose:
        print("Average distance: "+str(avg_dist))
        print("Bad lines:")
        print(bad_lines)
        print("Bad distances:")
        print([distances[i] for i in bad_inds])
#     lines=np.delete(lines,bad_inds,0)
    all_inds=[k for k in range(len(lines))]
    good_inds=list(set(all_inds)-set(bad_inds))
    return good_inds,bad_inds


# In[18]:


# sort a nx2 array of values by ascending x (n[0]) values
# can provide up to two lists to be sorted according to metric
def sort_by_asc(np_array_n_by_2,extra_list=None):
    global discon
    if discon: metric_xy=[math.pow(math.pow(np_array_n_by_2[i][0],2)+math.pow(np_array_n_by_2[i][1],2),1/2)
                        for i in range(len(np_array_n_by_2))]
    else: metric_xy=[np_array_n_by_2[i][0] for i in range(len(np_array_n_by_2))]
    if extra_list is not None: return map(np.array,zip(*sorted(zip(metric_xy,np_array_n_by_2, extra_list))))
    return np.asarray(list(map(list,list([x for _,x in sorted(zip(metric_xy,np_array_n_by_2))]))))


# In[19]:


def get_intersections(group0,group1):
    intersections=np.ndarray(shape=(len(group0),len(group1),2))
    for i in range(len(group0)):
        rho0=group0[i][0]
        theta0=group0[i][1]
        line0=np.array([math.cos(theta0),math.sin(theta0),-rho0])
        for j in range(len(group1)):
            rho1=group1[j][0]
            theta1=group1[j][1]
            line1=np.array([math.cos(theta1),math.sin(theta1),-rho1])
            intersection_point=np.cross(line0,line1)
            intersection_point=intersection_point/intersection_point[2]
            for k in [0,1]: intersections[i][j][k]=intersection_point[k]
        intersections[i]=sort_by_asc(intersections[i])
    if intersections[0][-1][1] < intersections[-1][0][1]: intersections=np.transpose(intersections,(1,0,2))
    return intersections


# In[20]:


def createReference(img_size,num_squares=8):
    square_size=img_size/num_squares
    return np.array([[[i, j] for j in range(9)] for i in range(9)])*square_size


# In[21]:


def grab_ortho(src_intersections,src_image):
    ortho_size=256
    ref_intersections=createReference(ortho_size)
    ref_corners=np.array([ref_intersections[0][0],ref_intersections[0][-1],
                          ref_intersections[-1][0],ref_intersections[-1][-1]])
    src_corners=np.array([src_intersections[0][0],src_intersections[0][-1],
                          src_intersections[-1][0],src_intersections[-1][-1]])
    h_image_ortho,_=cv2.findHomography(srcPoints=src_corners,dstPoints=ref_corners)
    h_ortho_image=np.linalg.inv(h_image_ortho)
    pruned_intersections=clean_intersections(src_intersections,ref_intersections,h_ortho_image)
    ortho_image = cv2.warpPerspective(src=src_image, M=h_image_ortho, dsize=(ortho_size, ortho_size))
    return ortho_image,pruned_intersections,h_image_ortho


# In[22]:


# shaping array from (n , m , 2)
# to (n x m , 2)
def reshape_array(array):
    new_array=np.reshape(a=array,newshape=(array.shape[0]*array.shape[1], 2))
    return np.insert(new_array, 2, 1, axis=1)


# In[23]:


# we cookin
def clean_intersections(intersections,ref_intersections,h_ortho_image):
    inters=reshape_array(intersections)
    ref_inters=np.array([np.matmul(h_ortho_image,i)/np.matmul(h_ortho_image,i)[2] for i in reshape_array(ref_intersections)])
    save_inters=np.array([[0,0,0] for i in range(ref_inters.shape[0])])
    # find least squares error
    for i in range(ref_inters.shape[0]):
        dists=np.array([1e9 for i in range(inters.shape[0])])
        for j in range(inters.shape[0]):
            dists[j]=np.linalg.norm(ref_inters[i]-inters[j])
        val,idx=min((val,idx) for (idx,val) in enumerate(dists))
        save_inters[i]=inters[idx]
    for i in range(save_inters.shape[0]):
        for j in range(save_inters.shape[0]):
            if i == j : continue
            distance=np.linalg.norm(save_inters[i]-save_inters[j])
            if distance < 10.0:
#                 print("USING REFERENCE INTERSECTIONS")
                return ref_inters
    return save_inters


# In[24]:


def draw_board(image,intersections,just_boundary=True,board_color=(0,255,0)):
    if not just_boundary:
        draw_intersections(image,intersections,just_corners=False,all_color=board_color)
    # drawing boundary clockwise starting from upper left
    line_thickness=3
    cv2.line(img=image, 
             pt1=(int(intersections[0][0]),int(intersections[0][1])), 
             pt2=(int(intersections[-9][0]),int(intersections[-9][1])), 
             color=board_color, thickness=line_thickness, lineType=8)
    cv2.line(img=image, 
             pt1=(int(intersections[-9][0]),int(intersections[-9][1])), 
             pt2=(int(intersections[-1][0]),int(intersections[-1][1])), 
             color=board_color, thickness=line_thickness, lineType=8)
    cv2.line(img=image, 
             pt1=(int(intersections[-1][0]),int(intersections[-1][1])), 
             pt2=(int(intersections[8][0]),int(intersections[8][1])), 
             color=board_color, thickness=line_thickness, lineType=8)  
    cv2.line(img=image, 
             pt1=(int(intersections[8][0]),int(intersections[8][1])), 
             pt2=(int(intersections[0][0]),int(intersections[0][1])), 
             color=board_color, thickness=line_thickness, lineType=8)  

def point_distance(p1,p2):
    return math.pow(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2),1/2)

# return true if board is good
# false otherwise
def check_board(intersections,h_image_ortho):
#    points=np.array([intersections[0],  # top left
#        intersections[-9],              # top right 
#        intersections[-1],              # bottom right
#        intersections[8]])              # bottom left
#    print(points)
#    points=np.array([np.matmul(h_image_ortho,i)/np.matmul(h_image_ortho,i)[2] for i in points])
#    print(points)
#    for i in range(4):
#        l1=point_distance(points[i%4],points[(i+1)%4])
#        l2=point_distance(points[(i+1)%4],points[(i+2)%4])
#        print(l1)
#        print(l2)
#        print(l1/l2)
#        if l1 / l2 > 1.1 or l1 / l2 < 0.9: return False
    return True


# In[25]:


def draw_intersections(image,intersections,just_corners=True,all_color=(0,255,0)):
    radius=5
    thickness=3
    colors=[(255,255,255),(255,0,255),(0,255,255),(160,160,160),all_color]
    if not just_corners:
        for b in intersections:
            color=colors[-1]
            cv2.circle(img=image, center=(int(b[0]),int(b[1])), 
                       radius=radius, color=color, thickness=thickness, lineType=8, shift=0) 
    else:
        #draw corner intersections
        cv2.circle(img=image, center=(int(intersections[0][0]),int(intersections[0][1])), 
               radius=radius, color=colors[0], thickness=thickness, lineType=8, shift=0)
        cv2.circle(img=image, center=(int(intersections[8][0]),int(intersections[8][1])), 
           radius=radius, color=colors[1], thickness=thickness, lineType=8, shift=0)
        cv2.circle(img=image, center=(int(intersections[-9][0]),int(intersections[-9][1])), 
           radius=radius, color=colors[2], thickness=thickness, lineType=8, shift=0)
        cv2.circle(img=image, center=(int(intersections[-1][0]),int(intersections[-1][1])), 
           radius=radius, color=colors[3], thickness=thickness, lineType=8, shift=0)


# In[26]:


def detect_board(bgr_image, do_draw_board=False):
    global discon
    # get hough lines for image
    lines=detect_lines(bgr_image)
#         print(lines)

    # get euclidean transform of lines
    scaled_lines=hough_to_euc(lines)#,verbose=True,tag="")

#         print(scaled_lines)
#         temp=scaled_lines.copy()
#         lineser.append(temp)
#         lineser=[lineser, temp]

    _,scaled_lines,lines=sort_by_asc(scaled_lines,lines)

    # group lines into 2 orthogonal groups
    clustered_labels=cluster_lines(scaled_lines,n_clusters=2)

    # get line groups
    group0 = lines[clustered_labels==0]
    group1 = lines[clustered_labels==1]

    group0g,group0b=get_l1_prune_indices(group0)#,verbose=True)
    group1g,group1b=get_l1_prune_indices(group1)#,verbose=True)

    intersections=get_intersections(group0[group0g],group1[group1g])

    ortho_image,intersections,h_image_ortho=grab_ortho(intersections,bgr_image)

    # draw all lines
#     draw_lines(lines[clustered_labels==0][group0g],bgr_image,color=(0,0,255))
#     draw_lines(lines[clustered_labels==0][group0b],bgr_image,color=(0,255,0))
#     draw_lines(lines[clustered_labels==1][group1g],bgr_image,color=(255,0,0))
#     draw_lines(lines[clustered_labels==1][group1b],bgr_image,color=(0,190,255))

# #         check first and last lines of groups
#     draw_lines([lines[clustered_labels==0][group0g][0]],bgr_image,color=(0,190,255))
#     draw_lines([lines[clustered_labels==0][group0g][-1]],bgr_image,color=(0,0,128))
#     draw_lines([lines[clustered_labels==1][group1g][0]],bgr_image,color=(255,255,0))
#     draw_lines([lines[clustered_labels==1][group1g][-1]],bgr_image,color=(0,0,0))

    if not check_board(intersections,h_image_ortho): return None, None

    if do_draw_board:
        board_color=(140,140,140)
        draw_board(bgr_image,intersections,just_boundary=False,board_color=board_color)
        draw_intersections(bgr_image,intersections,just_corners=False,all_color=board_color)

    # reset discon
    discon=False
    return ortho_image, intersections


# In[27]:


#def main():
#    # we run the program for all of the provided test images
#    for image in images:
#
#        file_name, file_extension = os.path.splitext(image)
#        print("Image : "+image)
#        bgr_image = cv2.imread(image)
#        ortho_image,intersections=detect_board(bgr_image,do_draw_board=True)
#        
#        display_plot(name="ortho",image=ortho_image,size=(10,10))
#        save_image(str(image+"_ortho"),ortho_image)
#        
#        display_plot(file_name,bgr_image)
#        save_image(str(image+"_lines"),bgr_image)
#
#
## In[28]:
#
#
#images=[images[-1]]
#main()


# ## saving images for model training

# In[29]:


def save_squares():
    for image in images:
        file_name, file_extension = os.path.splitext(image)
        base_name=os.path.basename(file_name)
        print("Image : "+image)
        bgr_image = cv2.imread(image)
        ortho_image,_=detect_board(bgr_image,do_draw_board=False)
        display_plot(name="ortho board",image=ortho_image,size=(10,10))
        size=ortho_image.shape[0]
        square_size=int(size/8)
        for i in range(8):
            for j in range(8):
                crop_img = ortho_image[j*square_size:j*square_size+square_size, i*square_size:i*square_size+square_size]
                display_plot(name=base_name+" square "+str(i)+" "+str(j),image=crop_img,size=(5,5))
                if os.path.isfile(str("../data/populated/"+base_name+"_square_"+format(i*8+j, '02d')+ext)) or os.path.isfile(str("../data/unpopulated/"+base_name+"_square_"+format(i*8+j, '02d')+ext)): 
                    print("File found, moving to next file.")
                    continue
                while True:
                    ans=input("Is this square populated? ")
                    if ans=="y": to_save="../data/populated/"
                    else: to_save="../data/unpopulated/"
                    if ans == "show": 
                        display_plot(name="ortho board",image=ortho_image,size=(10,10))
                        display_plot(name="square "+str(i)+" "+str(j),image=crop_img,size=(5,5))
                    if ans == "y" or ans == "n": break
                crop_name=str(to_save+base_name+"_square_"+format(i*8+j, '02d')+ext)
                cv2.imwrite(crop_name,crop_img)


# In[ ]:




