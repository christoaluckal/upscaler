# Image Metric Calculations. Just put the the folder directory in the correct order. Original|SRGAN|ISR|PIL. Or use the below command
import sys
from image_similarity_measures.quality_metrics import rmse,psnr,ssim,fsim,issm,sre,sam,uiq
import cpbd
import cv2
import os
from PIL import Image

# python3 image_metrics.py all_images/original/ all_images/downscaled_upscaled_SRGAN/ all_images/downscaled_upscaled_ISR/ all_images/PIL_upscale/

def siftdensity(img1):
#load images
    # Initiate SIFT detector
    # img1 = cv2.imread(img1)
    height,width = img1.shape
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    density = len(kp1)/(0.01*0.01*height*width)
    return density

# RGB Image Comparisons
def compute_metrics(og,alt):
    alt_rmse = rmse(og,alt) # Root Mean Square Error
    alt_psnr = psnr(og,alt) # Peak Signal-to-Noise Ratio (dB)
    alt_ssim = ssim(og,alt) # Structural Similarity Index
    og_gray = cv2.cvtColor(og,cv2.COLOR_BGR2GRAY)
    alt_gray = cv2.cvtColor(alt,cv2.COLOR_BGR2GRAY)
    og_density = siftdensity(og_gray) # No. of SIFT features per 100x100px
    alt_density = siftdensity(alt_gray)
    return (alt_rmse,alt_psnr,alt_ssim,og_density,alt_density)
    

args = sys.argv[1:]
og_images = args[0]
srgan_images = args[1]
isr_images = args[2]
pil_images = args[3]

result = open('img_metric.txt','w+') # Change this to whatever

og_img_list = sorted([x for x in os.listdir(og_images) if x.lower().endswith('.jpg')])
srgan_img_list = sorted([x for x in os.listdir(srgan_images) if x.lower().endswith('.jpg')])
isr_img_list = sorted([x for x in os.listdir(isr_images) if x.lower().endswith('.jpg')])
pil_img_list = sorted([x for x in os.listdir(pil_images) if x.lower().endswith('.jpg')])

# Average metric calculations
og_density = 0

srgan_rmse = 0
srgan_psnr = 0
srgan_ssim = 0
srgan_alt_density = 0

isr_rmse = 0
isr_psnr = 0
isr_ssim = 0
isr_alt_density = 0

pil_rmse = 0
pil_psnr = 0
pil_ssim = 0
pil_alt_density = 0

counter = 0

from random import randint as rd

# Make a zip list and randomnly take an image
zip_list = list(zip(og_img_list,srgan_img_list,isr_img_list,pil_img_list))
for i in range(0,10): # Change this to whatever length
    rand_int = rd(0,len(og_img_list)-1)
    a,b,c,d = zip_list[rand_int]
    # for a,b,c,d in zip_list:
    if a==b==c==d:
        img1 = cv2.imread(og_images+a)

        img2 = cv2.imread(srgan_images+b)
        img22 = cv2.resize(img2,(5472,3648))
        img2_metrics = compute_metrics(img1,img22)
        print(a,'| {} SRGAN:'.format(b),img2_metrics)
        srgan_rmse+=img2_metrics[0]
        srgan_psnr+=img2_metrics[1]
        srgan_ssim+=img2_metrics[2]
        srgan_alt_density+=img2_metrics[4]
        og_density+=img2_metrics[3]
        result.write(('{}|{}-SRGAN: RMSE:{} PSNR:{} SSIM:{} OG-SIFT:{} ALT-SIFT:{}\n'.format(a,b,img2_metrics[0],img2_metrics[1],img2_metrics[2],img2_metrics[3],img2_metrics[4])))

        img3 = cv2.imread(isr_images+c)
        img3_metrics = compute_metrics(img1,img3)
        print(a,'| {} ISR:'.format(c),img3_metrics)
        result.write(('{}|{}-ISR: RMSE:{} PSNR:{} SSIM:{} OG-SIFT:{} ALT-SIFT:{}\n'.format(a,c,img3_metrics[0],img3_metrics[1],img3_metrics[2],img3_metrics[3],img3_metrics[4])))
        isr_rmse+=img3_metrics[0]
        isr_psnr+=img3_metrics[1]
        isr_ssim+=img3_metrics[2]
        isr_alt_density+=img3_metrics[4]

        img4 = cv2.imread(pil_images+d)
        img4_metrics = compute_metrics(img1,img4)
        print(a,'| {} PIL:'.format(d),img4_metrics)
        result.write(('{}|{}-PIL: RMSE:{} PSNR:{} SSIM:{} OG-SIFT:{} ALT-SIFT:{}\n'.format(a,d,img4_metrics[0],img4_metrics[1],img4_metrics[2],img4_metrics[3],img4_metrics[4])))
        pil_rmse+=img4_metrics[0]
        pil_psnr+=img4_metrics[1]
        pil_ssim+=img4_metrics[2]
        pil_alt_density+=img4_metrics[4]
        
        print("---------------------------------------------------{}\n".format(counter))
        counter+=1

srgan_rmse=srgan_rmse/10
srgan_psnr=srgan_psnr/10
srgan_ssim=srgan_ssim/10
srgan_alt_density=srgan_alt_density/10

isr_rmse=isr_rmse/10
isr_psnr=isr_psnr/10
isr_ssim=isr_ssim/10
isr_alt_density=isr_alt_density/10

pil_rmse=pil_rmse/10
pil_psnr=pil_psnr/10
pil_ssim=pil_ssim/10
pil_alt_density=pil_alt_density/10

result.write("---------------------------------------------------\n")
result.write("OG SIFT DENSITY:{}\n".format(og_density/10))
result.write("SRGAN AVG RMSE:{} SRGAN AVG PSNR:{} SRGAN AVG SSIM:{} SRGAN AVG SIFT DENSITY PER 100x100:{}\n".format(srgan_rmse,srgan_psnr,srgan_ssim,srgan_alt_density))
result.write("ISR AVG RMSE:{} ISR AVG PSNR:{} ISR AVG SSIM:{} ISR AVG SIFT DENSITY PER 100x100:{}\n".format(isr_rmse,isr_psnr,isr_ssim,isr_alt_density))
result.write("PIL AVG RMSE:{} PIL AVG PSNR:{} PIL AVG SSIM:{} PIL AVG SIFT DENSITY PER 100x100:{}\n".format(pil_rmse,pil_psnr,pil_ssim,pil_alt_density))





