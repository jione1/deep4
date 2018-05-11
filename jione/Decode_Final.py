import tensorflow as tf
import glob
import os
IMAGE_PATH = "C:/Users/user/Desktop/ML/Section11/data/train/"

IMAGE_LIST = []
IMAGE_LABEL = []

for path, dirs, files in os.walk(IMAGE_PATH):
    data_path = os.path.join(path, "*.jpg")
    files = glob.glob(data_path)
    if files:
        label = os.path.dirname(os.path.realpath(files[0])).split("\\")[-1]
        for filename in files:
            IMAGE_LIST.append(filename)
            IMAGE_LABEL.append(label)

print(len(IMAGE_LIST))
print(len(IMAGE_LABEL))
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
image_queue = tf.train.string_input_producer(IMAGE_LIST)

reader_image = tf.WholeFileReader()

key, value = reader_image.read(image_queue)
tf.Print(value, [value])
image_decoded = tf.decode_raw(value, tf.uint8)

x = tf.cast(image_decoded, tf.float32)
print(x)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    image_tensor = sess.run([x])
    print(image_tensor)

    coord.request_stop()
    coord.join(threads)