
'''
-------------------------------------------
-    Author: Asadullah Dal                -
-    =============================        -
-    Company Name: AiPhile                -
-    =============================        -
-    Purpose : Youtube Channel            -
-    ============================         -
-    Link: https://youtube.com/c/aiphile  -
-------------------------------------------
'''
# import the necessary packages
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

ref_point = []
click = False
points =()
cap = cv.VideoCapture(1)
_, frame = cap.read()
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

lk_params = dict(winSize=(20, 20),
                 maxLevel=4,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
  
cap = cv.VideoCapture(1)
point_selected = False
points = [()]
old_points = np.array([[]])
qr_detected= False
# stop_code=False

frame_counter =0
starting_time =time.time()
# keep looping until the 'q' key is pressed
while True:
    frame_counter +=1
    ret, frame = cap.read()
    img = frame.copy()
    # img = cv.resize(img, None, fx=2, fy=2,interpolation=cv.INTER_CUBIC)
    cv.imshow('old frame ', old_gray)
    cv.imshow('img', img)

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # display the image and wait for a keypress
    clone = frame.copy()
    hull_points =detectQRcode(frame)
    # print(old_points.size)
    stop_code=False
    if hull_points:
        pt1, pt2, pt3, pt4 = hull_points
        qr_detected= True
        stop_code=True
        old_points = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)
        frame =AiPhile.fillPolyTrans(frame, hull_points, AiPhile.MAGENTA, 0.4)
        AiPhile.textBGoutline(frame, f'Detection: Pyzbar', (30,80), scaling=0.5,text_color=(AiPhile.MAGENTA ))
        cv.circle(frame, pt1, 3, AiPhile.GREEN, 3)
        cv.circle(frame, pt2, 3, (255, 0, 0), 3)
        cv.circle(frame, pt3, 3,AiPhile.YELLOW, 3)
        cv.circle(frame, pt4, 3, (0, 0, 255), 3)
    if qr_detected and stop_code==False:
        # print('detecting')
        new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_points = new_points 
        new_points=new_points.astype(int)
        n = (len(new_points))
        frame =AiPhile.fillPolyTrans(frame, new_points, AiPhile.GREEN, 0.4)
        AiPhile.textBGoutline(frame, f'Detection: Optical Flow', (30,80), scaling=0.5,text_color=AiPhile.GREEN)
        cv.circle(frame, (new_points[0]), 3,AiPhile.GREEN, 2)

    old_gray = gray_frame.copy()
    # press 'r' to reset the window
    key = cv.waitKey(1)
    if key == ord("s"):
        cv.imwrite(f'reference_img/Ref_img{frame_counter}.png', img)

    # if the 'c' key is pressed, break from the loop
    if key == ord("q"):
        break
    fps = frame_counter/(time.time()-starting_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("image", frame)

# close all open windows
cv.destroyAllWindows()
cap.release()