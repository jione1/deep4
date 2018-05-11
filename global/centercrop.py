from PIL import Image
import os

# 이 부분만 수정
root = './food/carrot'
image_size = 128
save_root = './food/carrot_128'
prefix = 'g' # 구글이면 g 바이두면 b 붙이자

filenames = os.listdir(root)

for filename in (filenames):
    full_filename = os.path.join(root, filename)
    img = Image.open(full_filename)
    img = img.convert('RGB')

    shorter_side = min(img.size)
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2

    img = img.crop((half_the_width - shorter_side / 2,
                half_the_height - shorter_side / 2,
                half_the_width + shorter_side / 2,
                half_the_height + shorter_side / 2
                ))
    img = img.resize((image_size, image_size))

    if not os.path.exists(save_root):
        os.mkdir(save_root)
    img.save(save_root + '/' + prefix + filename.split('.')[0] + '.jpg')