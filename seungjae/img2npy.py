import numpy as np
from PIL import Image
import os
import glob

def one_hot(i):
    a = np.zeros(15, 'uint8')
    a[i] = 1
    return a

data_dir = './train/'
nb_classes = 15

result_arr = np.empty((12432, 64*64*3 + nb_classes)) # (전체 이미지 갯수, 64x64x3 + 15(클래스 갯수))
food_names = os.listdir(data_dir) # carrot, cheese, egg, ...

idx_start = 0
for cls, food in enumerate(food_names):
    file_list = glob.glob(data_dir + food + '/*.jpg')
    print(file_list)
    print(len(file_list))

    for idx, f in enumerate(file_list):
        im = Image.open(f)
        pix = np.array(im)
        arr = pix.reshape(1, 64*64*3)
        result_arr[idx_start + idx] = np.append(arr, one_hot(cls))
    idx_start += len(file_list)

np.save('result.npy', result_arr)

# split train/test
np.random.shuffle(result_arr)
training_data = result_arr[:10000, :]
test_data = result_arr[10000:, :]

np.save('training_data.npy', training_data)
np.save('test_data.npy', test_data)