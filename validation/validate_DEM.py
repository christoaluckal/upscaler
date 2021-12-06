og_dem = '/home/caluckal/Desktop/Github/elevation-infer/validation/original_DEM.tif'
altered_dem = '/home/caluckal/Desktop/Github/elevation-infer/validation/downscaled_upscaled_DEM.tif'

from osgeo import gdal
import numpy as np
import cv2
from sklearn.preprocessing import normalize

def getdata(path):
    og_height,og_width= cv2.imread(path,-1).shape
    dem_file = gdal.Open(str(path))
    dem_band = dem_file.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray(0,0,og_width,og_height)
    return dem_data,og_height,og_width
    # cv2.imwrite('/home/caluckal/Desktop/Github/elevation-infer/validation/test.png',dem_data)

def getfirstvalid(array,height,width):
    try:
        for x in range(height):
            for y in range(width):
                if array[x][y] == -32767.0:
                    print(array[x][y])
                    raise Exception
    except Exception:
        return (x,y)

og_data,og_height,og_width = getdata(og_dem)
alt_data,alt_height,alt_width = getdata(altered_dem)

box_list_sel = []

def draw_rectangle_with_drag(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        box_list_sel.append((4*x,4*y))
        print("Clicked on {},{} which is {},{} in original sub-image".format(x,y,4*x,4*y))


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


og_img = cv2.imread('validation/og.png')
alt_img = cv2.imread('validation/alt.png')


x_dash,y_dash = tri_sel(og_img,alt_img)

print(x_dash,y_dash)

dummy = np.zeros((alt_height,alt_width))


def min_no_outlier(data,height,width):
    min = 0
    for x in range(height):
        for y in range(width):
            test = data[x][y]
            if test < min and test!=-32767:
                min = test
    return min

def offset_data(og,dummy,y_off,x_off):
    for y in range(og_height):
        for x in range(og_width):
            if og_data[y][x] != -32767:
                dummy[y+y_off][x+x_off] = og[y][x]
    
    return dummy

dummy = offset_data(og_data,dummy,y_dash,x_dash)
min_og = min_no_outlier(og_data,og_height,og_width)
max_og = np.max(og_data)


def normalize(x,min,max,a,b):
    norm = int(a+(b-a)*((x-min)/(max-min)))
    return norm

dummy_img = np.zeros((alt_height,alt_width,3),dtype=np.int32)
diff = dummy-alt_data
for x in range(alt_height):
    for y in range(alt_width):
        if diff[x][y]!=-32767:
            norm = normalize(diff[x][y],min_og,max_og,-128,127)
            color = np.array([min(255,2*abs(norm)),min(255,abs((255-norm)*2)),0])
            print(color)
        else:
            color = np.array([0,0,0])
        dummy_img[x][y] = color
        
        

cv2.imwrite('validation/color_test.png',dummy_img)

# from sklearn.metrics import mean_squared_error as mse
# for x in diff:
#     zero = np.zeros((len(x)))
#     print(mse(zero,x))

# diff = np.array(255*diff).astype(np.int8)
# mses = ((diff)**2).mean(axis=1)
# cv2.imwrite('validation/proper_testing.jpg',diff)


