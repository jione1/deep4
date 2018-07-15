import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read();

    print(ret)
    print(frame)
    frame = cv2.flip(frame, 1) # 좌우반전(거울모드)
    cv2.imshow('image', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
