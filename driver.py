import os

from numpy.core.fromnumeric import resize
import split
import join
import glob
import time
import sys
import cv2

args = sys.argv[1:]

done = open('done.txt','a')

def clear_dir(dirname):
    files = glob.glob(dirname+'*')
    for f in files:
        os.remove(f)

def modify_image(original_folder,original_list,type):
    for x in original_list:
        image = cv2.imread(original_folder+x)
        height,width,_ = image.shape
        if type=='resize':
            modified = cv2.resize(image,(width//4,height//4),interpolation = cv2.INTER_AREA)
            cv2.imwrite(original_folder+'resized_'+x,modified)
    # pass


def run_upscaler():
    from subprocess import Popen
    os.chdir('Fast-SRGAN/')
    # Popen('cd Fast-SRGAN/',shell=True)
    process = Popen('python3 infer.py --image_dir ../split_output/ --output_dir ../split_upscaled/',shell=True)
    process.wait()
    os.chdir('../')
    return

original_folder = args[0]
output_folder = args[1]

image_list = os.listdir(original_folder)
# image_list = modify_image(original_folder,image_list,'resize')

for x in image_list:
    start = time.time()
    # Split the larger image into smaller 96x96 images
    vsteps,hsteps = split.split_96(original_folder+x,output_folder)
    # This function runs the inference code and generates the 384x384 upscaled image for the corresponding 96x96 image
    run_upscaler()
    # The upscaler scaled all 96x96 images into 384x384. Since they all belong to one original image, we stitch the 384x384 image to make the larger image
    join.join_96(vsteps,hsteps,'/home/caluckal/Desktop/temp/split_upscaled','/home/caluckal/Desktop/temp/upscaled')
    # Delete the contents of the temp folder
    clear_dir(output_folder)
    clear_dir('split_upscaled/')
    end = time.time()
    print("FINISHED:",x," in {}s".format(end-start))
    done.write((x+'\n'))
