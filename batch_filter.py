import sys
import os
import numpy as np
import cv2
args = sys.argv[1:]

folder = args[0]

image_list = [x for x in os.listdir(folder) if x.endswith('.jpg') or x.endswith('.png')]

def sharpen(image_name):
    image = cv2.imread(image_name)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    modified = cv2.filter2D(image, -1, kernel)
    print(image[0][0],modified[0][0])
    cv2.imwrite(image_name[:-4]+'_new.jpg',modified)


for x in image_list:
    sharpen(folder+x)