import cv2

def split_96(image,output):
    color_ortho = cv2.imread(image)
    image_name = image.split('/')[-1][:-4]
    vert_steps,horr_steps,_ = color_ortho.shape

    vert_steps = vert_steps//96
    horr_steps = horr_steps//96

    for x in range(vert_steps):
        for y in range(horr_steps):
            # print(96*x,96*x+96,96*y,96*y+96)
            temp_image = color_ortho[96*x:96*x+96,96*y:96*y+96]
            # print(output+image_name+"_{}_{}_.jpg".format(x,y))
            cv2.imwrite(output+image_name+'_{}_{}_.jpg'.format(x,y),temp_image)

    return vert_steps,horr_steps
