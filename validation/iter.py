# This code is used to iterate 2 dems over each other and print the RMSE,PSNR,SSIM metrics for each offset
from osgeo import gdal
import numpy as np
import cv2
from sklearn.preprocessing import normalize
import math

import sys

# Calculate which is the larger DEM. The larger DEM is kept stationary and the smaller DEM is offseted around the larger
def getdata(dem1_path,dem2_path):
    dem1_height,dem1_width = cv2.imread(dem1_path,-1).shape
    dem1_file = gdal.Open(str(dem1_path))
    dem1_band = dem1_file.GetRasterBand(1)
    dem1_data = dem1_band.ReadAsArray(0,0,dem1_width,dem1_height)

    dem2_height,dem2_width = cv2.imread(dem2_path,-1).shape
    dem2_file = gdal.Open(str(dem2_path))
    dem2_band = dem2_file.GetRasterBand(1)
    dem2_data = dem2_band.ReadAsArray(0,0,dem2_width,dem2_height)

    if dem2_width > dem1_width:
        print("DEM2 bigger")
        big_dem = dem2_data
        big_width = dem2_width
        big_height = dem2_height
        small_dem = dem1_data
        small_width = dem1_width
        small_height = dem1_height
    else:
        print("DEM1 bigger")
        big_dem = dem1_data
        big_width = dem1_width
        big_height = dem1_height
        small_dem = dem2_data
        small_width = dem2_width
        small_height = dem2_height       

    return (big_dem,big_height,big_width),(small_dem,small_height,small_width)
    # cv2.imwrite('/home/caluckal/Desktop/Github/elevation-infer/validation/test.png',dem_data)

box_list_sel = []
range_px = []
scale = 1

def draw_rectangle_with_drag(event, x, y, flags, param):
    global scale
    if event == cv2.EVENT_LBUTTONDOWN:
        box_list_sel.append((scale*x,scale*y))
        print(scale*x,scale*y)

def selector(image_og,scale):
    '''
    Function to display the image with contours and select and normalize the clicked coordinates

    Params
    image_og: Image to be displayed

    Returns
    box_list_sel: 2D array with the coordinates (X,Y) of the clicked location
    '''
    # Pixel location variables
    img_og_shape = image_og.shape
    # print(img_og_shape)
    # We downscale the original image to be able to show it in a window. This definitely leads to a ~5% error in pixel calculations
    img_disp = cv2.resize(image_og,(img_og_shape[1]//scale,img_og_shape[0]//scale))
    # Pixel location storage    
    cv2.namedWindow(winname = "Downscaled Sub-Image")
    cv2.setMouseCallback("Downscaled Sub-Image", 
                        draw_rectangle_with_drag)
    img_disp = cv2.cvtColor(img_disp,cv2.COLOR_BGR2RGB)
    #cv2.imwrite('contour_op_selection.jpg',img_disp)
    while True:
        cv2.imshow("Downscaled Sub-Image", img_disp)
        
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()

    return box_list_sel

def tri_sel(image1,image2):
    '''
    Takes image1 and image2 and displays it on screen allowing user to click on it. selector method takes scale as second parameter
    1 = Full size
    2 = Half size
    4 = Quarter size
    '''
    count = 0
    global scale
    scale = 1
    while count<3:
        selector(image1,1)
        selector(image2,1)
        count+=1

    x_avg = (box_list_sel[1][0]-box_list_sel[0][0])+(box_list_sel[3][0]-box_list_sel[2][0])+(box_list_sel[5][0]-box_list_sel[4][0])
    x_avg = x_avg//3

    y_avg = (box_list_sel[1][1]-box_list_sel[0][1])+(box_list_sel[3][1]-box_list_sel[2][1])+(box_list_sel[5][1]-box_list_sel[4][1])
    y_avg = y_avg//3

    scale = 4

    return (x_avg,y_avg)


def offset_data(og,dummy,y_off,x_off):
    '''
    The smaller DEM is put into an array of the same size as the larger DEM and offseted by some x,y
    '''
    og_height,og_width = og.shape
    d_height,d_width = dummy.shape
    if og_height+y_off < d_height and og_width+x_off < d_width:
        for y in range(og_height):
            for x in range(og_width):
                dummy[y+y_off][x+x_off] = og[y][x]
    
        return dummy
    else:
        return None



def RMSE(array1,array2):
    from sklearn.metrics import mean_squared_error as mse
    mse_val = mse(array1,array2)
    rmse = math.sqrt(mse_val)
    return rmse

def PSNR(array1,rmse):
    psnr = 20*math.log10(np.max(array1)/rmse)
    return psnr


def flat(array1,array2,range_list):
    '''
    Flattens the DEM array to the specified range and removes all instances of either DEM having -32767 as a value
    '''
    x_min,y_min = 140 ,140
    x_max,y_max = 3400 ,3000
    flat1 = []
    flat2 = []
    for y in range(y_min,y_max):
        for x in range(x_min,x_max):
            if (array1[y][x]!=-32767) and (array2[y][x]!=-32767):
                flat1.append(array1[y][x])
                flat2.append(array2[y][x])
            else:
                pass
    return flat1,flat2

def get_min(array):
    '''
    Getting minimum excluding the outlier
    '''
    min_t = 0
    for x in range(0,len(array)):
        for y in range(0,len(array[0])):
            val = array[x][y]
            # print(val)
            if val!=-32767:
               if val < min_t:
                    min_t = array[x][y]

    return min_t

# Predefined values to speed up calculation

og_max = 28.364176
og_min = -21.73187

srgan_max = 24.3338
srgan_min = -16.420786
srgan_f_max = 24.362696
srgan_f_min = -29.01959

isr_max = 24.542622
isr_min = -22.41958
isr_f_max = 24.612194
isr_f_min = -16.995506

pil_max = 24.781492
pil_min = -14.546541
pil_f_max = 24.410984 
pil_f_min = -21.677523

avg_max = 24.41634
avg_min = -14.85847

def make_image(array,name,out,og_flag):
    '''
    Make a normalized image of the array. og_flag is for if the image is the original DEM and not a generated one
    '''
    new_arr = np.array(array)
    if og_flag == False:
        if dem_type == 'SRGAN':
            max_val = srgan_max
            min_val = srgan_min
        elif dem_type == 'ISR':
            max_val = isr_max
            min_val = isr_min
        elif dem_type == 'AVG':
            max_val = avg_max
            min_val = avg_min
        elif dem_type == 'PIL':
            max_val = pil_max
            min_val = pil_min
        elif dem_type == 'PIL_F':
            max_val = pil_f_max
            min_val = pil_f_min
        elif dem_type == 'ISR_F':
            max_val = isr_f_max
            min_val = isr_f_min
        elif dem_type == 'SRGAN_F':
            max_val = srgan_f_max
            min_val = srgan_f_min
    else:
        if dem_type == 'TEST':
            max_val = isr_max
            min_val = isr_min
        else:
            max_val = og_max
            min_val = og_min

    height,width = new_arr.shape
    for x in range(height):
        for y in range(width):
            if new_arr[x][y]!=-32767:
                new_arr[x][y] = int((new_arr[x][y]-min_val)*255/(max_val-min_val))
            else:
                new_arr[x][y] = 0
    image_array = new_arr
    cv2.imwrite(out+name, image_array)

    # cv2.imwrite(out+name, new_arr)


def SSIM(img,img_noise,min_v,max_v):
    from skimage.metrics import structural_similarity as ssim
    ssim_noise = ssim(img, img_noise,
                  data_range=max_v - min_v)
    return ssim_noise

def savenpy(array,text):
    with open('iteration_testing/{}/{}.npy'.format(dem_type,text),'wb') as f:
        np.save(f,array)

def loadnpy(loc):
    arr = np.load(loc)
    return arr

def gen_data(dem1,dem2):
    '''
    Saving the DEM arrays to speed up calculation. Use this method first before calculating metrics
    '''
    print("DEM_TYPE:",dem_type)
    big_data,small_data = getdata(dem1,dem2)
    big_dem_data,big_dem_height,big_dem_width = big_data
    small_dem_data,small_dem_height,small_dem_width = small_data
    make_image(big_dem_data,'iteration_testing/{}/BIG_DEM.png'.format(dem_type),'',False)
    # make_image(small_dem_data,'iteration_testing/{}/SMALL_DEM.png'.format(dem_type),'',False)
    savenpy(big_dem_data,'BIG_DEM')
    savenpy(small_dem_data,'SMALL_DEM')


from time import time
def dem_metrics():
    '''
    Metrics calculations. The method prints the metric values. Set the offset range to whatever you desire.
    (0,50,5) means the offset will lie in that range
    '''
    big_dem_data = loadnpy('iteration_testing/{}/BIG_DEM.npy'.format(dem_type))
    small_dem_data = loadnpy('iteration_testing/{}/SMALL_DEM.npy'.format(dem_type))
    big_image = cv2.imread('iteration_testing/{}/BIG_DEM.png'.format(dem_type))
    big_dem_height,big_dem_width = big_dem_data.shape
    for i in range(y_start,y_end,y_step):
        for j in range(x_start,x_end,x_step):
            start = time()
            offset = np.zeros((big_dem_height,big_dem_width))
            offseted = offset_data(small_dem_data,offset,i,j)
            if offseted is not None:
                make_image(offseted,'iteration_testing/{}/OFFSET.png'.format(dem_type),'',True)
                offset_img = cv2.imread('iteration_testing/{}/OFFSET.png'.format(dem_type))
                flat_dem_1,flat_dem_2 = flat(big_dem_data,offseted,range_px)   
                rmse = RMSE(flat_dem_1,flat_dem_2)
                print("RMSE:",rmse)       
                psnr_max = np.max(flat_dem_2)
                psnr_score = PSNR(flat_dem_2,rmse)
                print("PSNR:",psnr_max,' ',psnr_score)
                x_min,y_min = 140 ,140
                x_max,y_max = 3400 ,3000
                image_1 = big_image[y_min:y_max,x_min:x_max]
                image_1 = cv2.split(image_1)[0]
                image_2 = offset_img[y_min:y_max,x_min:x_max]
                image_2 = cv2.split(image_2)[0]
                max_val = 255
                min_val = 0
                ssim_score = SSIM(image_2,image_1,min_val,max_val)
                print("SSIM:",ssim_score)
                res.write((str(i)+','+str(j)+":|{},{},{},{}|\n".format(rmse,psnr_max,psnr_score,ssim_score)))
                end = time()
                print("{},{} PER ITER:{}".format(i,j,end-start))

args = sys.argv[1:]
dem1 = args[0]
dem2 = args[1]
dem_type = args[2]
gen_flag = int(args[3])
if int(dem_type) == 0:
    dem_type = 'SRGAN'
elif int(dem_type) == 1:
    dem_type = 'SRGAN_F'
elif int(dem_type) == 2:
    dem_type = 'ISR'
elif int(dem_type) == 3:
    dem_type = 'ISR_F'
elif int(dem_type) == 4:
    dem_type = 'PIL'
elif int(dem_type)==5:
    dem_type = 'PIL_F'
else:
    dem_type = 'AVG_F'

# Large Checking
y_start,y_end,y_step = 0,10,1
x_start,x_end,x_step = 0,10,1

# Small Checking
# Take the best (y,x) written in results.txt and subtract and add 5 to the ranges
# For a best RMSE at 10,5
# y_start,y_end,y_step = 5,15,1
# x_start,x_end,x_step = 0,10,1


if gen_flag==1:
    gen_data(dem1,dem2) # USE THIS FIRST BEFORE METRICS
else:
    res = open('iteration_testing/{}/results.txt'.format(dem_type),'w+')
    dem_metrics()

# SRGAN y:27 x:6
# ISR y:5 x:2
# AVG offset (SRGAN>ISR) (y:22 x:4)
# AVG y:27 x:6 
# PIL: y:8 x:5
# PIL_F: y:8 x:5
