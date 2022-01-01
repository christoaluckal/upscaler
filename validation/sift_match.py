
import os
import cv2

def match(img1,img2,name):
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
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
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

# good_list = []
# import sys
# args = sys.argv[1:]

# og_folder = args[0]
# alt_folder = args[1]
# match_folder = args[2]

# # og_folder = '/home/caluckal/Desktop/Github/upscaler/validation/IP_testing/ISR/sift_results/og_parts'
# # alt_folder = '/home/caluckal/Desktop/Github/upscaler/validation/IP_testing/ISR/sift_results/alt_parts'



# og_list = sorted([x for x in os.listdir(og_folder) if x.endswith('.jpg')])
# alt_list = sorted([x for x in os.listdir(alt_folder) if x.endswith('.jpg')])

# match_list = []
# name_list = []
# for x in og_list[0:10]:
#     # print(x)
#     names = x.split('_')
#     name = names[2]+'_'+names[3]
#     print(name)
#     if 'alt_Ortho_'+name+'_.jpg' in alt_list:
#         match_list.append([x,'alt_Ortho_'+name+'_.jpg'])

# for x in match_list[0:1]:
#     names = x[0].split('_')
#     name = names[2]+'_'+names[3]
#     good_list.append(match(og_folder+x[0],alt_folder+x[1],match_folder+name))
#     name_list.append(name)

# match_file = open(match_folder+'match_list.txt','w+')
# for x,y in zip(good_list,name_list):
#     if x is not None:
#         print(y,':',len(x))
#         match_file.write((str(y)+'.jpg  :'+str(len(x))+'\n'))