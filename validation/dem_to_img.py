import cv2
import sys
from osgeo import gdal
import numpy as np
args = sys.argv[1:]

dem_file = args[0]
out_name = args[1]

# Read DEM TIFF and save as basic grayscale image

def get_dem_image(dem1_path):
    dem1_height,dem1_width = cv2.imread(dem1_path,-1).shape
    dem1_file = gdal.Open(str(dem1_path))
    dem1_band = dem1_file.GetRasterBand(1)
    dem1_data = dem1_band.ReadAsArray(0,0,dem1_width,dem1_height)

    return dem1_data


dem_data = get_dem_image(dem_file)
dem_data = np.array(dem_data).astype(np.int32)
dem_image = cv2.imwrite(out_name+'.png',dem_data)