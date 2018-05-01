import numpy as np
from PIL import Image
import os

def one_hot(i):
    a = np.zeros(15, 'uint8')
    a[i] = 1
    return a

data_dir = './train/'
nb_classes = 15

result_arr = np.empty((12432, 12303)) # (전체 이미지 갯수, 64x64x3 + 15(클래스 갯수))

food_list = os.listdir(data_dir)
idx_start = 0
for cls, food_name in enumerate(food_list):
    image_dir = data_dir + food_name + '/'
    jpg_list = os.listdir(image_dir)
    for idx, f in enumerate(jpg_list):
        im = Image.open(image_dir + f)
        pix = np.array(im)
        arr = pix.reshape(1, 12288)
        result_arr[idx_start + idx] = np.append(arr, one_hot(cls))
    idx_start += len(file_list)

np.save('result.npy', result_arr)