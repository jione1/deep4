import cv2
import os
import glob

# Get user supplied values
file_names = glob.glob('./son3/*.png') + glob.glob('./son3/*.jpg')
print(file_names)

for f in file_names:

    cascPath = 'haarcascade_frontalface_default.xml'

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(f)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print(f)
    print("Found {0} faces!".format(len(faces)))

    if len(faces) == 1:
        # crop face
        for (x, y, w, h) in faces:
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            crop_img = image[y: y + h, x: x + w]  # Crop from x, y, w, h -> 100, 200, 300, 400

        crop_img = cv2.resize(crop_img, (64,64), interpolation=cv2.INTER_AREA)
        cv2.imwrite('son_crop/d%s.jpg' % f[-10:-4], crop_img)
    else:
        pass