import cv2
import numpy as np
im = cv2.imread('difference.png')


count = [0,0,0,0,0,0,0,0,0]

def np_cmp(array1,array2):
    if((array1==np.array(array2)).all()):
        return True
    else:
        return False

for y in range(im.shape[0]):
    print(y)
    for x in range(im.shape[1]):
        dif = im[y][x]
        if not np_cmp(dif,[255,255,255]):
            if np_cmp(dif,[3,4,122]):
                count[0]+=1
            elif np_cmp(dif,[5,47,208]):
                count[1]+=1
            elif np_cmp(dif,[33,126,251]):
                count[2]+=1
            elif np_cmp(dif,[58,207,238]):
                count[3]+=1
            elif np_cmp(dif,[60,252,164]):
                count[4]+=1
            elif np_cmp(dif,[152,242,50]):
                count[5]+=1
            elif np_cmp(dif,[235,188,40]):
                count[6]+=1
            elif np_cmp(dif,[227,107,70]):
                count[7]+=1
            elif np_cmp(dif,[59,18,48]):
                count[8]+=1

print(count)


# if dif > 1:
#     color = np.array([3,4,122])
# elif dif > 0.75:
#     color = np.array([5,47,208])
# elif dif > 0.5:
#     color = np.array([33,126,251])
# elif dif > 0.2:
#     color = np.array([58,207,238])
# elif dif > 0.1:
#     color = np.array([60,252,164])
# elif dif > 0.075:
#     color = np.array([152,242,50])
# elif dif > 0.05:
#     color = np.array([235,188,40])
# elif dif > 0.02:
#     color = np.array([227,107,70])
# else:
#     color = np.array([59,18,48])