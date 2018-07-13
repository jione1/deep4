import os
import shutil

img_dir = './face'

save_dir = './train'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

idx = 0
for dir in os.listdir(img_dir):
    for img in os.listdir(img_dir + '/' + dir):

        shutil.copy('%s/%s/%s' % (img_dir, dir, img),
                    '%s/train_%05d.jpg' % (save_dir, idx))
        idx += 1