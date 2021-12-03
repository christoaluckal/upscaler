# upscaler

The motivation behind this code is to test if Orthomosaics,DEMs,DTMs generated through an AI upscaled image will provide the same (if not more) detail compared to a regular aerial image.

1.  Simply run as: `python3 driver.py <input_folder>`
2.  This code will create 3 folders which are used for temporary storage. The upscaled image will be stored in a folder called `upscaled_big`
3.  From there, if you want GPS data to be copied simply run `python3 exif_test.py <exif_image_folder> <upscaled_image_folder>` and gps metadata will be copied


### To DO ###
1.  [X] Call the SRGAN code through a method and not a subprocess
2.  Modify/Train the SRGAN model for custom images
