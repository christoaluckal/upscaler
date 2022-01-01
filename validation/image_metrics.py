import sys
from image_similarity_measures.quality_metrics import rmse,psnr,ssim,fsim,issm,sre,sam,uiq
import cpbd
import cv2
import os
from PIL import Image

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

def compute_metrics(og,alt):
    alt_rmse = rmse(og,alt)
    alt_psnr = psnr(og,alt)
    alt_ssim = ssim(og,alt)
    og_gray = cv2.cvtColor(og,cv2.COLOR_BGR2GRAY)
    alt_gray = cv2.cvtColor(alt,cv2.COLOR_BGR2GRAY)
    og_density = siftdensity(og_gray)
    alt_density = siftdensity(alt_gray)
    return (alt_rmse,alt_psnr,alt_ssim,og_density,alt_density)
    

args = sys.argv[1:]
og_images = args[0]
srgan_images = args[1]
isr_images = args[2]
pil_images = args[3]

result = open('img_metric.txt','w+')

og_img_list = sorted([x for x in os.listdir(og_images) if x.lower().endswith('.jpg')])
srgan_img_list = sorted([x for x in os.listdir(srgan_images) if x.lower().endswith('.jpg')])
isr_img_list = sorted([x for x in os.listdir(isr_images) if x.lower().endswith('.jpg')])
pil_img_list = sorted([x for x in os.listdir(pil_images) if x.lower().endswith('.jpg')])


counter = 0
for a,b,c,d in list(zip(og_img_list,srgan_img_list,isr_img_list,pil_img_list)):
    if a==b==c==d:
        img1 = cv2.imread(og_images+a)

        img2 = cv2.imread(srgan_images+b)
        img22 = cv2.resize(img2,(5472,3648))
        img2_metrics = compute_metrics(img1,img22)
        print(a,'| {} SRGAN:'.format(b),img2_metrics)
        result.write(('{}|{}-SRGAN: RMSE:{} PSNR:{} SSIM:{} OG-SIFT:{} ALT-SIFT:{}\n'.format(a,b,img2_metrics[0],img2_metrics[1],img2_metrics[2],img2_metrics[3],img2_metrics[4])))

        img3 = cv2.imread(isr_images+c)
        img3_metrics = compute_metrics(img1,img3)
        print(a,'| {} ISR:'.format(c),img3_metrics)
        result.write(('{}|{}-ISR: RMSE:{} PSNR:{} SSIM:{} OG-SIFT:{} ALT-SIFT:{}\n'.format(a,c,img3_metrics[0],img3_metrics[1],img3_metrics[2],img3_metrics[3],img2_metrics[4])))

        img4 = cv2.imread(pil_images+d)
        img4_metrics = compute_metrics(img1,img4)
        print(a,'| {} PIL:'.format(d),img4_metrics)
        result.write(('{}|{}-PIL: RMSE:{} PSNR:{} SSIM:{} OG-SIFT:{} ALT-SIFT:{}\n'.format(a,d,img4_metrics[0],img4_metrics[1],img4_metrics[2],img4_metrics[3],img2_metrics[4])))
        print("---------------------------------------------------{}\n".format(counter))
        counter+=1
    result.write("---------------------------------------------------\n")





