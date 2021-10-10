import AiPhile
import cv2 as cv 
import numpy as np
import time 

cap = cv.VideoCapture(1)

frame_counter =0
start_time =time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter +=1
    fps = frame_counter/(time.time() - start_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps, 2)}', (30,40), scaling=0.8)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key==ord('q'):
        break
cv.destroyAllWindows()
cap.release()