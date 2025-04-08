# import os
# # path_m = '28_2_103380.png'
# path_m = "original_dataset\/club_01_ace\\uk_60min_1_frame_4260.jpg"
# path_basename = os.path.basename(path_m)
# print(path_basename)
# print(path_m.replace('.png', '.jpg').split('.jpg'))
# file_name = path_m.replace('.png', '.jpg').split('.jpg')[1] + f"prediction_abcd" + ".jpg"
# print(file_name)

import os

arr = os.listdir('original_dataset\/club_06/')
print(arr)

# os.remove("original_dataset\/club_06\/desktop.ini")