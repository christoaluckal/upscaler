from osgeo import gdal
import numpy as np
import cv2
from sklearn.preprocessing import normalize
import math

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
        print(x,y)

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
    print(img_og_shape)
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


def min_no_outlier(data,height,width):
    min = 0
    for x in range(height):
        for y in range(width):
            test = data[x][y]
            if test < min and test!=-32767:
                min = test
    return min

def offset_data(og,dummy,y_off,x_off):
    og_height,og_width = og.shape
    for y in range(og_height):
        for x in range(og_width):
            dummy[y+y_off][x+x_off] = og[y][x]
    
    return dummy

def normalize(x,min,max,a,b):
    norm = int(a+(b-a)*((x-min)/(max-min)))
    return norm



# %%
def get_difference(array1,array2,height,width):
    image = np.zeros((height,width,3),dtype=np.int32)
    difference = np.zeros((height,width,1),dtype=np.float64)
    for x in range(height):
        for y in range(width):
            if(array1[x][y]== -32767 and array2[x][y] == -32767):
                color = np.array([0,0,0])
                difference[x][y] = 0
            else:
                dif =  abs(array1[x][y]-array2[x][y])
                if dif > 1:
                    color = np.array([0,0,255])
                elif dif > 0.5:
                    color = np.array([0,138,255])
                elif dif > 0.2:
                    color = np.array([0,255,255])
                elif dif > 0.1:
                    color = np.array([0,255,138])
                else:
                    color = np.array([0,255,0])
                difference[x][y] = dif
            image[x][y]=color
            
    return image,difference


def RMSE(array1,array2):
    from sklearn.metrics import mean_squared_error as mse
    # sum = 0
    # for x in range(0,len(array1)):
    #     sum = sum + pow((array1[x]-array2[x]),2)
    # rmse = math.sqrt(sum/len(array1))
    mse_val = mse(array1,array2)
    rmse = math.sqrt(mse_val)
    return rmse

def PSNR(array1,rmse):
    psnr = 20*math.log10(np.max(array1)/rmse)
    return psnr

def flat(array1,array2,range_list):
    x_min,y_min = range_list[0][0],range_list[0][1]
    x_max,y_max = range_list[1][0],range_list[1][1]
    flat1 = []
    flat2 = []
    for y in range(y_min,y_max):
        for x in range(x_min,x_max):
            if (array1[y][x]!=-32767) and (array2[y][x]!=-32767):
                flat1.append(array1[y][x])
                flat2.append(array2[y][x])
            else:
                pass
            #     flat1.append(big)
            # if (array2[y][x]!=-32767):
            #     flat2.append(array2[y][x])
            # else:
            #     flat2.append(big)
    return flat1,flat2

def sflat(array1,range_list):
    x_min,y_min = range_list[0][0],range_list[0][1]
    x_max,y_max = range_list[1][0],range_list[1][1]
    flat1 = []
    for y in range(y_min,y_max):
        for x in range(x_min,x_max):
            if (array1[y][x]!=-32767):
                flat1.append(array1[y][x])
            else:
                pass
            #     flat1.append(big)
            # if (array2[y][x]!=-32767):
            #     flat2.append(array2[y][x])
            # else:
            #     flat2.append(big)
    return flat1

import sys
args = sys.argv[1:]
dem1 = args[0]
dem2 = args[1]

def get_min(array):
    min_t = 0
    for x in range(0,len(array)):
        for y in range(0,len(array[0])):
            val = array[x][y]
            # print(val)
            if val!=-32767:
               if val < min_t:
                    min_t = array[x][y]

    return min_t

def make_image(array,name,out):
    new_arr = np.array(array)
    # max_val = np.max(new_arr)
    # min_val = get_min(new_arr)
    # height,width = new_arr.shape
    # print(height,width,max_val,min_val)
    # for x in range(height):
    #     for y in range(width):
    #         if new_arr[x][y]!=-32767:
    #             new_arr[x][y] = int((new_arr[x][y]-min_val)*255/(max_val-min_val))
    #         else:
    #             new_arr[x][y] = 0
    # image_array = np.array(new_arr).astype(np.int8)
    cv2.imwrite(out+name, new_arr)



def SSIM(img,img_noise,min_v,max_v):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img_noise = cv2.cvtColor(img_noise,cv2.COLOR_BGR2GRAY)

    from skimage.metrics import structural_similarity as ssim
    ssim_noise = ssim(img, img_noise,
                  data_range=max_v - min_v)
    return ssim_noise

def validate_dems(dem1,dem2):
    big_data,small_data = getdata(dem1,dem2)
    big_dem_data,big_dem_height,big_dem_width = big_data
    small_dem_data,small_dem_height,small_dem_width = small_data
    make_image(big_dem_data,'BIG_DEM.png','')
    make_image(small_dem_data,'SMALL_DEM.png','')
    big_image = cv2.imread('BIG_DEM.png')
    small_image = cv2.imread('SMALL_DEM.png')
    x_dash,y_dash = tri_sel(small_image,big_image)
    # x_dash,y_dash = 8,30
    print(x_dash,y_dash)
    selector(big_image,4)
    selector(big_image,4)
    range_px.append(box_list_sel[-2])
    range_px.append(box_list_sel[-1])
    offset = np.zeros((big_dem_height,big_dem_width))
    offseted = offset_data(small_dem_data,offset,y_dash,x_dash)
    make_image(offseted,'OFFSET.png','')
    offset_img = cv2.imread('OFFSET.png')
    # image_diff,diff_array = get_difference(big_dem_data,offseted,big_dem_height,big_dem_width)

    flat_dem_1,flat_dem_2 = flat(big_dem_data,offseted,range_px)
    
    rmse = RMSE(flat_dem_1,flat_dem_2)
    print("RMSE:",rmse)

    print("PSNR:",np.max(flat_dem_2),PSNR(flat_dem_1,rmse))

    max_val = np.max(flat_dem_2)
    min_val = np.min(flat_dem_2)

    image_1 = big_image[range_px[0][1]:range_px[1][1],range_px[0][0]:range_px[1][0]]
    image_1 = cv2.split(image_1)[0]
    image_2 = offset_img[range_px[0][1]:range_px[1][1],range_px[0][0]:range_px[1][0]]
    image_2 = cv2.split(image_2)[0]

    max_val = 255
    min_val = 0
    print("SSIM:",SSIM(image_2,image_1,min_val,max_val))


    import pandas as pd

    pd_1 = pd.DataFrame(flat_dem_1)
    pd_2 = pd.DataFrame(sflat(offseted,range_px))

    print("ALTERED")
    print(pd_1.describe().apply(lambda s: s.apply('{0:.5f}'.format)))
    print("ORIGINAL")
    print(pd_2.describe().apply(lambda s: s.apply('{0:.5f}'.format))
)
    # cv2.imwrite('bracketed_save.png',image_diff)


validate_dems(dem1,dem2)