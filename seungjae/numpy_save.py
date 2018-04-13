import numpy as np
from PIL import Image
import os

Gothic_list = ['NanumGothic', 'KBIZHanmaeumGothic M', 'HANYoonGothic740', 'aChamGothicM']

for font_name in Gothic_list:
    path_dir = 'image/Gothic/' + font_name + '/'
    file_list = os.listdir(path_dir)

    result_arr = np.empty((2350, 1804))
    for idx, f in enumerate(file_list):
        im = Image.open(path_dir + f)
        im = im.convert('L')
        pix = np.array(im)
        arr = pix.reshape(1, 1800)
        result_arr[idx] = np.append(arr, [1, 0, 0, 0])

    np.save('npy/' + font_name + '.npy', result_arr)