# upscaler

The motivation behind this code is to test if Orthomosaics,DEMs,DTMs generated through an AI upscaled image will provide the same (if not more) detail compared to a regular aerial image.

1.  Simply run as: `python3 driver.py <input_folder>`
2.  This code will create 3 folders which are used for temporary storage. The upscaled image will be stored in a folder called `upscaled_big`
3.  From there, if you want GPS data to be copied simply run `python3 exif_test.py <exif_image_folder> <upscaled_image_folder>` and gps metadata will be copied


### To DO ###
1.  [X] Call the SRGAN code through a method and not a subprocess
2.  Modify/Train the SRGAN model for custom images

## Citation ###
Check out the results here <br>
`@INPROCEEDINGS{9884486,
  author={Turkar, Yash and Aluckal, Christo and De, Shaunak and Turkar, Varsha and Agarwadkar, Yogesh},
  booktitle={IGARSS 2022 - 2022 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Generative-Network Based Multimedia Super-Resolution for Uav Remote Sensing}, 
  year={2022},
  volume={},
  number={},
  pages={527-530},
  doi={10.1109/IGARSS46834.2022.9884486}}
`
