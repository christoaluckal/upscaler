import os
import numpy as np
import split_image
import join_chunks
import glob
import time
import sys
import cv2
import isr_infer

args = sys.argv[1:]


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
            cv2.imwrite('modded/'+'{}_'.format(type)+x,modified)
        elif type=='blur':
            modified = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
            cv2.imwrite('modded/'+'{}_'.format(type)+x,modified)
        elif type=='sharpen':
            kernel = np.array([[-1,-1,-1], [-1,3,-1], [-1,-1,-1]])
            modified = cv2.filter2D(image, -1, kernel)
            cv2.imwrite('modded/'+'{}_'.format(type)+x,modified)

    # pass


def run_upscaler(split_output,upscaled_output):
    from subprocess import Popen
    os.chdir('Fast-SRGAN/')
    # Popen('cd Fast-SRGAN/',shell=True)
    process = Popen('python3 infer.py --image_dir ../{}/ --output_dir ../{}/'.format(split_output,upscaled_output),shell=True)
    process.wait()
    os.chdir('../')
    return

def create_temp_folders(flag):
    # if not os.path.exists('split_output'):
    #     os.mkdir('./split_output')
    # else:
    #     clear_dir('split_output/')
    # if not os.path.exists('split_upscaled'):
    #     os.mkdir('./split_upscaled')
    # else:
    #     clear_dir('split_upscaled/')
    if not os.path.exists('upscaled_big'):
        os.mkdir('./upscaled_big')
    else:
        clear_dir('upscaled_big/')
    if flag is True:
        if not os.path.exists('modded'):
            os.mkdir('./modded')


original_folder = args[0]
try:
    modification = args[1]
    create_temp_folders(True)
except Exception:
    modification = None
    create_temp_folders(False)
# output_folder = args[1]


image_list = [x for x in os.listdir(original_folder) if x.lower().endswith('.jpg')]
if modification is not None:
    modify_image(original_folder,image_list,modification)
    image_list = os.listdir('./modded')
    original_folder = 'modded/'
start = time.time()

for x in image_list:
    isr_infer.upscale(x,original_folder,'upscaled_big')
    print("DONE WITH: {}".format(x))

end = time.time()
print("\n\n\n")
print("FINISHED {} files in {}s".format(len(image_list),end-start))
