import os
import sys
import cv2

args = sys.argv[1:]
folder = args[0]
# out_folder = args[1]

keypoints = 0
def sift(image,name):
    global keypoints
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    keypoints+=len(kp)
    img=cv2.drawKeypoints(gray,kp,image)
    # cv2.imwrite(name[:-4]+'_new.jpg',img)


img_list = [x for x in os.listdir(folder) if x.endswith('.jpg')]

for x in img_list:
    image = cv2.imread(folder+'/'+x)
    # sift(image,out_folder+x)
    sift(image)

print(keypoints)

