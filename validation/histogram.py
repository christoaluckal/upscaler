import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
args = sys.argv[1:]
# og_array = np.load(args[0])
srgan_array = np.load(args[0])
sr_off = np.load(args[1])
isr_array = np.load(args[2])
isr_off = np.load(args[3])
avg_array = np.load(args[4])
avg_off = np.load(args[5])

# big_array_h,big_array_w = big_array.shape

def get_difference(array1,array2,height,width):
    difference = np.zeros((height,width),dtype=np.float64)
    for x in range(height):
        for y in range(width):
            if(array1[x][y]== -32767 or array2[x][y] == -32767):
                difference[x][y] = 0
            else:
                dif =  array1[x][y]-array2[x][y]
                difference[x][y] = dif
            
    return difference

def offset_data(og,dummy,y_off,x_off):
    og_height,og_width = og.shape
    for y in range(og_height):
        for x in range(og_width):
            dummy[y+y_off][x+x_off] = og[y][x]
    
    return dummy

def RMSE(array1,array2):
    from sklearn.metrics import mean_squared_error as mse
    mse_val = mse(array1,array2)
    rmse = math.sqrt(mse_val)
    return rmse

def ME(array1,array2):
    length = len(array1)
    error = 0
    for x in range(length):
        error = error+array1[x]-array2[x]
    error = error/len(array1)
    return error

def flat(array1,array2):
    x_min,y_min = 140,140
    x_max,y_max = 3400,3000
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

def flat2(array1):
    x_min,y_min = 140,140
    x_max,y_max = 3400,3000
    flat1 = []
    for y in range(y_min,y_max):
        for x in range(x_min,x_max):
            if array1[y][x] !=-32767:
                flat1.append(array1[y][x])
    return np.array(flat1)

# offset = np.zeros((big_array_h,big_array_w))
# offseted = offset_data(sm_array,offset,27,6)


# with open('iteration_testing/AVG/OFFSETED.npy','wb') as f:
#         np.save(f,offseted)



# flat_srg,flat_og_1 = flat(srgan_array,sr_off)
# flat_isr,flat_og_2 = flat(isr_array,isr_off)
# flat_avg,flat_og_3 = flat(avg_array,avg_off)


# diff_1 = np.array(flat_srg)-np.array(flat_og_1)
# diff_2 = np.array(flat_isr)-np.array(flat_og_2)
# diff_3 = np.array(flat_avg)-np.array(flat_og_3)

# with open('flats/srgan_diff.npy','wb') as f:
#         np.save(f,diff_1)

# with open('flats/isr_diff.npy','wb') as f:
#         np.save(f,diff_2)

# with open('flats/avg_diff.npy','wb') as f:
#         np.save(f,diff_3)

diff_1 = np.load('flats/srgan_diff.npy')
diff_2 =  np.load('flats/isr_diff.npy')
diff_3 = np.load('flats/avg_diff.npy')




# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# print(colors)
from random import uniform
for x in range(1,20):
    rand = uniform(0,1)
    range_1,size = int(uniform(0.75,1)*600000),200
    print(range_1,size)
    diff_1_sm = diff_1[range_1:range_1+size]
    diff_2_sm = diff_2[range_1:range_1+size]
    diff_3_sm = diff_3[range_1:range_1+size]

    diff_1_mean = np.mean(diff_1_sm)
    diff_2_mean = np.mean(diff_2_sm)
    diff_3_mean = np.mean(diff_3_sm)

    plt.hist([diff_1_sm,diff_2_sm,diff_3_sm],bins=10,label=['srgan','isr','avg'])
    plt.axvline(x=diff_1_mean,color='#1f77b4')
    plt.axvline(x=diff_2_mean,color='#ff7f0e')
    plt.axvline(x=diff_3_mean,color='#2ca02c')
    plt.legend()
    # plt.xticks((-3,-2,-1,0,1,2,3))
    plt.show()