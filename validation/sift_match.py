
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

good_list = []

og_folder = '/home/caluckal/Desktop/Github/upscaler/validation/IP_testing/ISR/sift_results/og_parts'
alt_folder = '/home/caluckal/Desktop/Github/upscaler/validation/IP_testing/ISR/sift_results/alt_parts'



og_list = sorted([x for x in os.listdir(og_folder) if x.endswith('.jpg')])
alt_list = sorted([x for x in os.listdir(alt_folder) if x.endswith('.jpg')])

match_list = []
name_list = []
for x in og_list:
    names = x.split('_')
    name = names[2]+'_'+names[3]
    if 'alt_parts_'+name+'_.jpg' in alt_list:
        match_list.append([x,'alt_parts_'+name+'_.jpg'])

for x in match_list:
    names = x[0].split('_')
    name = names[2]+'_'+names[3]
    good_list.append(match(og_folder+'/'+x[0],alt_folder+'/'+x[1],'/home/caluckal/Desktop/Github/upscaler/validation/IP_testing/ISR/sift_results/match/'+name))
    name_list.append(name)

match_file = open('/home/caluckal/Desktop/Github/upscaler/validation/IP_testing/ISR/sift_results/match_list.txt','w+')


for x,y in zip(good_list,name_list):
    if x is not None:
        print(y,':',len(x))
        match_file.write((str(y)+'.jpg  :'+str(len(x))+'\n'))