# import the necessary packages
import cv2
import time

vs = cv2.VideoCapture("videos/day2.avi")

# allow the camera to warm up
time.sleep(2.0)

# keep looping
frameCounter = 1
while True:
    # grab the current frame
    ret, frame = vs.read()
    if not ret:
        break

    if frameCounter % 80 == 0:
        fileName = f'images/to_label/day2_{frameCounter:05d}.jpg'
        print(f'Writing {fileName}')
        cv2.imwrite(fileName, frame)

    frameCounter += 1

vs.release()
