import cv2
import os
import glob

image_path = './aoa'
save_path = './aoa_save'

# Get user supplied values
file_names = glob.glob(image_path + '/*.png') + glob.glob(image_path + '/*.jpg')
print(file_names)

for f in file_names:

    facePath = 'haarcascade_frontalface_default.xml'

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(facePath)

    # Read the image
    image = cv2.imread(f)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print(f)
    print("Found {0} face(s)!".format(len(faces)))
    print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imwrite('%s/%s' % (save_path, f.split('\\')[1]) , image)
    #else:
        #pass