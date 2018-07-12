import cv2
import os
import glob

image_path = './images'
save_path = './images_detected'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get user supplied values
file_names = glob.glob(image_path + '/*.png') + glob.glob(image_path + '/*.jpg')
print(file_names)

for f in file_names:

    facePath = 'haarcascade_frontalface_default.xml'
    smilePath = 'haarcascade_smile.xml'

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(facePath)
    smileCascade = cv2.CascadeClassifier(smilePath)

    # Read the image
    image = cv2.imread(f)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print(f)
    print("Found {0} face(s)!".format(len(faces)))
    print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.16,
            minNeighbors=40,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Set region of interest for smiles
        for (x2, y2, w2, h2) in smile:
            print("Found {0} smile(s)!".format(len(smile)))
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
            cv2.putText(image, 'Smile', (x, y - 7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite('%s/%s' % (save_path, f.split('\\')[1]), image)

