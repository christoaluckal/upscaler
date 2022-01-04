
import cv2
import numpy as np
from PIL import Image
import sys

args = sys.argv[1:]
image = args[0] # Image on which Corner-Edge detection is to be run
res_folder = args[1] # Result folder 
basename = args[2] # Base name for the results

cv_img = cv2.imread(image)

#Corner Detection
def corner(image,name,result_folder):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    image[dst>0.01*dst.max()]=[0,0,255]
    print("Writing:{}+harris_{}.png".format(result_folder,name))
    cv2.imwrite(result_folder+'harris_{}.png'.format(name),image)


# Edge Detection
def edge(img,name,result_folder):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    print("Writing:{}+sobelX_{}.png".format(result_folder,name))
    cv2.imwrite(result_folder+'sobelX_{}.png'.format(name),sobelx)
    
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    print("Writing:{}+sobelY_{}.png".format(result_folder,name))
    cv2.imwrite(result_folder+'sobelY_{}.png'.format(name),sobely)
    
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    print("Writing:{}+sobelXY_{}.png".format(result_folder,name))
    cv2.imwrite(result_folder+'sobelXY_{}.png'.format(name),sobelxy)
    
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # Display Canny Edge Detection Image
    print("Writing:{}+canny_{}.png".format(result_folder,name))
    cv2.imwrite(result_folder+'canny_{}.png'.format(name),edges)


corner(cv_img,basename,res_folder)
edge(cv_img,basename,res_folder)