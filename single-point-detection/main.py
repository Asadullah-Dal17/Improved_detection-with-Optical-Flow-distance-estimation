import numpy as np
import cv2 as cv 
import AiPhile
import time
def selectPoint(event, x , y , flags, parmas):
    global point, condition, old_point 
    if event ==cv.EVENT_LBUTTONUP:
        point =(int(x), int(y))
        old_point = np.array([[x, y]], dtype=np.float32)
        condition =True 

cv.namedWindow('frame')
cv.setMouseCallback('frame', selectPoint)

cap = cv.VideoCapture(1)
ret, frame = cap.read()
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

lk_param = dict(winSize=(20,20), maxLevel=4, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))

frame_counter=0
point =()
condition = False
old_point =np.array([[]])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_counter += 1
    fps = frame_counter/(time.perf_counter())
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,2)}', (40,30), scaling=0.8)
    if condition:
        cv.circle(frame, point, 5, AiPhile.MAGENTA, -1, cv.LINE_AA)

        new_point, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_point, None,  **lk_param)
        old_point = new_point
        new_point = new_point.astype(int)
        x, y = new_point.ravel()
        cv.circle(frame, (x,y), 5, AiPhile.GREEN, 2, cv.LINE_4)

    old_gray = gray_frame.copy() 

    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()