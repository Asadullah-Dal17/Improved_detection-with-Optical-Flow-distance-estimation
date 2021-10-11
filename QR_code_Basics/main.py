# imports 
import cv2 as cv 
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
import time
import AiPhile 

# QR code detector function 

def detectQRcode(image):
    # convert the color image to gray scale image
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode: 
        x, y, w, h =obDecoded.rect
        # cv.rectangle(image, (x,y), (x+w, y+h), ORANGE, 4)
        points = obDecoded.polygon
        if len(points) > 4:
            hull = cv.convexHull(
                np.array([points for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
  
        return hull

cap = cv.VideoCapture(0)
frame_counter =0
starting_time =time.time()
# keep looping until the 'q' key is pressed
while True:
    frame_counter +=1
    ret, frame = cap.read()
    hull_points =detectQRcode(frame)
    if hull_points:
        pt1, pt2, pt3, pt4 = hull_points
        frame =AiPhile.fillPolyTrans(frame, hull_points, AiPhile.MAGENTA, 0.6)
        AiPhile.textBGoutline(frame, f'Detection: Pyzbar', (30,80), scaling=0.5,text_color=(AiPhile.MAGENTA ))
        cv.circle(frame, pt1, 3, AiPhile.GREEN, 3)
        cv.circle(frame, pt2, 3, (255, 0, 0), 3)
        cv.circle(frame, pt3, 3,AiPhile.YELLOW, 3)
        cv.circle(frame, pt4, 3, (0, 0, 255), 3)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    fps = frame_counter/(time.time()-starting_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("image", frame)
# close all open windows
cv.destroyAllWindows() 
cap.release()