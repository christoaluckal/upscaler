import sys
import os
import numpy as np
import cv2
args = sys.argv[1:]
folder = args[0]
out_folder = args[1]
image_list = [x for x in os.listdir(folder) if x.lower().endswith('.jpg') or x.lower().endswith('.png')]

def sharpen(image_name):
    image = cv2.imread(image_name)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    modified = cv2.filter2D(image, -1, kernel)
    print(image[0][0],modified[0][0])
    cv2.imwrite(image_name[:-4]+'_new.jpg',modified)

def downscale(folder,image,factor,out):
    if not os.path.isfile(out+image):
        og = cv2.imread(folder+image)
        height,width,_ = og.shape
        new_height,new_width = height//factor,width//factor
        new = cv2.resize(og,(new_width,new_height),interpolation=cv2.INTER_AREA)
        cv2.imwrite(out+image,new)


for x in image_list:
    # sharpen(folder+x)
    downscale(folder,x,2,out_folder)