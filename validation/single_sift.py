
import os
import cv2

def match(img1,img2,name,factor):
#load images
    # Initiate SIFT detector
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    try:
        matches = bf.knnMatch(des1,des2, k=2)
    except Exception:
        return
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < factor*n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    print(name+'.jpg')
    cv2.imwrite(name+'.jpg',img3)
    return good

def siftdensity(img1):
#load images
    # Initiate SIFT detector
    img1 = cv2.imread(img1)
    height,width = img1.shape
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)

    density = len(kp1)/(height*width)
    return density

good_list = []
import sys
args = sys.argv[1:]

og_file = args[0]
alt_file = args[1]
op_file = args[2]
factor = float(args[3]) # Default 0.75

match(og_file,alt_file,op_file,factor)

