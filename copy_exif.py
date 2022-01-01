import piexif
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
import time
import sys

args = sys.argv[1:]

# dict_keys(['0th', 'Exif', 'GPS', 'Interop', '1st', 'thumbnail'])
def copy_exif(exif_image_path,non_exif_image_path):
    img_1 = Image.open(exif_image_path)
    exif_dict_1 = piexif.load(img_1.info['exif'])
    # print(exif_dict_1['GPS'][piexif.ImageIFD.DateTime])
    # print(exif_dict_1['GPS'])
    img_2 = Image.open(non_exif_image_path)
    temp_gps = exif_dict_1['GPS']
    exif_dict_2 = {"GPS":temp_gps}
    exif_2_bytes = piexif.dump(exif_dict_2)
    img_2.save(non_exif_image_path,exif=exif_2_bytes)


exif_loc = args[0]
noexif_loc = args[1]

import os
exif_list = sorted([exif_loc+x for x in os.listdir(exif_loc)])
noexif_list = sorted([noexif_loc+x for x in os.listdir(noexif_loc)])

start = time.time()
# count = 1
for x,y in zip(exif_list,noexif_list):
    file_x = x.split('/')[-1]
    file_y = y.split('/')[-1]
    if file_x[0:8] == file_y[0:8]:
        # print(count)
        # print(file_x,file_y)
        # count+=1
        copy_exif(x,y)
        # print("Done with: ",file_x)

end = time.time()

print("Total time for Exif copying of {} files is {}s".format(len(exif_list),(end-start)))