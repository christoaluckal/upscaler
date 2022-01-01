from ISR.models import RDN
import numpy as np
from PIL import Image
import os
def upscale(image,infolder,outfolder):
    img = Image.open(infolder+image)
    lr_img = np.array(img)
    rdn = RDN(weights='psnr-small')
    sr_img = rdn.predict(lr_img,by_patch_of_size=96)
    im = Image.fromarray(sr_img)
    im.save(outfolder+'/'+image)