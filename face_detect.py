import cv2
import os
import glob

# Get user supplied values
file_names = glob.glob('./aoa/*.png') + glob.glob('./aoa/*.jpg')
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
        minSize=(16, 16),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print(f)
    print("Found {0} faces!".format(len(faces)))
    print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imwrite('./aoa/box.jpg', image)
    #else:
        #pass