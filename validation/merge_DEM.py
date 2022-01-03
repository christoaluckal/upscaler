from osgeo import gdal
import numpy as np
import cv2

def get_dem_data(dem1_path,dem2_path):
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
        big_file = dem2_file
        small_dem = dem1_data
        small_width = dem1_width
        small_height = dem1_height
    else:
        print("DEM1 bigger")
        big_dem = dem1_data
        big_width = dem1_width
        big_height = dem1_height
        big_file = dem1_file
        small_dem = dem2_data
        small_width = dem2_width
        small_height = dem2_height       

    return (big_dem,big_height,big_width),(small_dem,small_height,small_width),big_file
    # cv2.imwrite('/home/caluckal/Desktop/Github/elevation-infer/validation/test.png',dem_data)

def get_first_valid(array,height,width):
    try:
        for x in range(height):
            for y in range(width):
                if array[x][y] == -32767.0:
                    print(array[x][y])
                    raise Exception
    except Exception:
        return (x,y)

box_list_sel = []

def draw_rectangle_with_drag(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        box_list_sel.append((4*x,4*y))
        print("selected")

def selector(image_og):
    '''
    Function to display the image with contours and select and normalize the clicked coordinates

    Params
    image_og: Image to be displayed

    Returns
    box_list_sel: 2D array with the coordinates (X,Y) of the clicked location
    '''
    # Pixel location variables
    img_og_shape = image_og.shape
    # We downscale the original image to be able to show it in a window. This definitely leads to a ~5% error in pixel calculations
    img_disp = cv2.resize(image_og,(img_og_shape[1]//4,img_og_shape[0]//4))
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
    while count<3:
        selector(image1)
        selector(image2)
        count+=1

    x_avg = (box_list_sel[1][0]-box_list_sel[0][0])+(box_list_sel[3][0]-box_list_sel[2][0])+(box_list_sel[5][0]-box_list_sel[4][0])
    x_avg = x_avg//3

    y_avg = (box_list_sel[1][1]-box_list_sel[0][1])+(box_list_sel[3][1]-box_list_sel[2][1])+(box_list_sel[5][1]-box_list_sel[4][1])
    y_avg = y_avg//3

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

def write_tiff(arr_out,height,width,ds,merged_loc):
    '''
    Write a numpy array to a TIFF file
    '''
    print(arr_out.shape)
    if merged_loc[-1] == '/':
        merged_loc = merged_loc[:-1]
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create("{}/merged.tif".format(merged_loc), width, height, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr_out)
    outdata.GetRasterBand(1).SetNoDataValue(-32767)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None


def get_difference(array1,array2,height,width):
    '''
    Make a difference array. If either values of the array are -32767 then the maximum of them are taken
    Otherwise average. May be a better way
    '''
    difference = np.zeros((height,width),dtype=np.float32)
    for x in range(height):
        for y in range(width):
            if array1[x][y] == -32767 or array2[x][y]== -32767:
                difference[x][y] = max(array1[x][y],array2[x][y])
            else:
                difference[x][y] = (array1[x][y]+array2[x][y])/2
                # difference[x][y] = min(array1[x][y],array2[x][y])

            
    return difference

def make_image(image_array,name,out):
    image_array = np.array(image_array).astype(np.int8)
    cv2.imwrite(out+name,image_array)

def delete_file(file):
    import os
    os.remove(file)
import math

def RMSE(array1,array2):
    from sklearn.metrics import mean_squared_error as mse
    mse_val = mse(array1,array2)
    rmse = math.sqrt(mse_val)
    return rmse

def PSNR(array1,rmse):
    psnr = 20*math.log10(np.max(array1)/rmse)
    return psnr

def flatten_dem_arrays(array1,array2,range_list):
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

range_px = []

def compute_merge(dem1,dem2,merged_loc):
    '''
    You can select the two DEMs and determine the optimal x_dash,y_dash. Or you can set it to values you want
    '''
    big_data,small_data,band_data = get_dem_data(dem1,dem2)
    big_dem_data,big_dem_height,big_dem_width = big_data
    small_dem_data,small_dem_height,small_dem_width = small_data
    make_image(big_dem_data,'BIG_DEM.png','')
    make_image(small_dem_data,'SMALL_DEM.png','')
    big_image = cv2.imread('BIG_DEM.png')
    small_image = cv2.imread('SMALL_DEM.png')
    x_dash,y_dash = tri_sel(small_image,big_image)
    # print(x_dash,y_dash)
    # x_dash,y_dash = 4,22 # CHANGE THIS HERE TO WHATEVER YOU WANT INCASE YOU DONT WANT TO SELECT
    offset = np.zeros((big_dem_height,big_dem_width))
    offseted = offset_data(small_dem_data,offset,y_dash,x_dash)
    make_image(offseted,'OFFSET.png','')
    diff_array = get_difference(big_dem_data,offseted,big_dem_height,big_dem_width)
    x_min,y_min = 140 ,140
    x_max,y_max = 3400 ,3000
    # range_px.append(box_list_sel[-2])
    # range_px.append(box_list_sel[-1])
    range_px.append([x_min,y_min])
    range_px.append([x_max,y_max])
    flat1,flat2 = flatten_dem_arrays(big_dem_data,offseted,range_px)
    print("RMSE:",RMSE(flat1,flat2))
    # cv2.imwrite('bracketed_save.png',image_diff)
    write_tiff(diff_array,big_dem_height,big_dem_width,band_data,merged_loc)
    delete_file('BIG_DEM.png')
    delete_file('SMALL_DEM.png')
    delete_file('OFFSET.png')


import sys
args = sys.argv[1:]
dem1 = args[0]
dem2 = args[1]
merged_out = args[2]


compute_merge(dem1,dem2,merged_out)
# %%
