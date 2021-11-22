import cv2
import numpy as np
import os

def join_96(steps_v,steps_h,data_dir,output_upscaled):
    blank = np.zeros((384*steps_v,384*steps_h,3), np.uint8)
    image_list = sorted(os.listdir(data_dir))
    image_name_split = image_list[0].split('_')
    # Get new image name
    image_name = "_".join(str(x) for x in image_name_split[0:-3])+"_upscaled"+image_name_split[-1]
    image_map = {}

    for images in image_list:
        num1 = images.split('_')[-3]
        num2 = images.split('_')[-2]
        # Save image name with the image chunk coords being the keys
        # image_map["1_2"] means the chunk on the 2nd row and 3rd column
        image_map[num1+"_"+num2] = images

    # print(image_map)

    for y in range(0,steps_v):
        for x in range(0,steps_h):
            # print(data_dir+'/'+image_map["{}_{}".format(y,x)])
            blank[384*y:384*(y+1),384*x:384*(x+1)] = cv2.imread(data_dir+'/'+image_map["{}_{}".format(y,x)])

    cv2.imwrite(output_upscaled+'/'+image_name,blank)