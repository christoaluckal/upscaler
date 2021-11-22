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


def run_upscaler(split_output,upscaled_output):
    from subprocess import Popen
    os.chdir('Fast-SRGAN/')
    # Popen('cd Fast-SRGAN/',shell=True)
    process = Popen('python3 infer.py --image_dir ../{}/ --output_dir ../{}/'.format(split_output,upscaled_output),shell=True)
    process.wait()
    os.chdir('../')
    return

def create_temp_folders():
    if not os.path.exists('split_output'):
        os.mkdir('./split_output')
    else:
        clear_dir('split_output/')
    if not os.path.exists('split_upscaled'):
        os.mkdir('./split_upscaled')
    else:
        clear_dir('split_upscaled/')
    if not os.path.exists('upscaled_big'):
        os.mkdir('./upscaled_big')
    else:
        clear_dir('upscaled_big/')


original_folder = args[0]
# output_folder = args[1]

image_list = os.listdir(original_folder)
# image_list = modify_image(original_folder,image_list,'resize')
create_temp_folders()

for x in image_list:
    start = time.time()
    # Split the larger image into smaller 96x96 images
    vsteps,hsteps = split.split_96(original_folder+x,'split_output/')
    # This function runs the inference code and generates the 384x384 upscaled image for the corresponding 96x96 image
    run_upscaler('split_output','split_upscaled')
    # The upscaler scaled all 96x96 images into 384x384. Since they all belong to one original image, we stitch the 384x384 image to make the larger image
    join.join_96(vsteps,hsteps,'split_upscaled','upscaled_big')
    # Delete the contents of the temp folder
    clear_dir('split_output/')
    clear_dir('split_upscaled/')
    end = time.time()
    print("FINISHED:",x," in {}s".format(end-start))
    done.write((x+'\n'))
