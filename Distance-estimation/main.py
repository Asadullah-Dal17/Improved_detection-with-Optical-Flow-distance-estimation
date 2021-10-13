
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
import math
import AiPhile
# important variables for distance Estimation. 
KNOWN_WIDTH = 1.8 #centimeters
KNOWN_DISTANCE = 15.5 # centimeters 

# Find the Distance between to points 
def eucaldainDistance(x, y, x1, y1):

    eucaldainDist = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    return eucaldainDist

# distance estimation function
def focalLength(measured_distance, real_width, width_in_rf_image):

    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length
    
def distanceFinder(Focal_Length, real_face_width, face_width_in_frame):

    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance

# QR code detector function 

def detectQRcode(image):
    # convert the color image to gray scale image
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode: 
        x, y, w, h =obDecoded.rect
        cv.rectangle(image, (x,y), (x+w, y+h), AiPhile.ORANGE, 4)
        points = obDecoded.polygon
  
        if len(points) > 4:
            hull = cv.convexHull(
                np.array([points for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        n = len(hull)

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

reference_image = cv.imread('../reference_img/Ref_img180.png')
ref_point = detectQRcode(reference_image)
if ref_point:
    print('detect Qr code in reference image')
    x, x1 = ref_point[0][0], ref_point[1][0]
    y, y1 = ref_point[0][1], ref_point[1][1]
    ref_height = eucaldainDistance(x, y, x1, y1)
    cv.line(reference_image, (x-10,y), (x-10, y+int(ref_height)), AiPhile.RED, 3, cv.LINE_AA)
    
    AiPhile.textBGoutline(reference_image, f'height/width: {round(ref_height, 3)}', (30,40), scaling=0.6)
    focal_length = focalLength(KNOWN_DISTANCE, KNOWN_WIDTH,ref_height)
    AiPhile.textBGoutline(reference_image, f'Focal_length: {round(focal_length,3)}', (30,80), bg_color=AiPhile.PURPLE, scaling=0.6)
    cv.line(reference_image, (x,y), (x1, y1), AiPhile.GREEN, 3, cv.LINE_AA)

else:
    print('QR code is not, in reference image')
cv.imshow('Reference image', reference_image)

frame_counter =0
starting_time =time.time()
# keep looping until the 'q' key is pressed
while True:
    frame_counter +=1
    ret, frame = cap.read()
    cv.imshow('old gray frame ', old_gray)

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    clone = frame.copy()
    hull_points =detectQRcode(frame)
    # print(old_points.size)
    stop_code=False
    if hull_points:
        pt1, pt2, pt3, pt4 = hull_points
        x, y = pt1
        x1, y1 = pt2
        eucaldain_dist = eucaldainDistance(x, y, x1, y1) #height or width of qr code 
        distance = distanceFinder(focal_length, KNOWN_WIDTH, eucaldain_dist)
        AiPhile.textBGoutline(frame, f'Distance: {round(distance,2)} cm', (340,50), scaling=0.8, text_color=AiPhile.MAGENTA )
        frame =AiPhile.fillPolyTrans(frame, hull_points, AiPhile.MAGENTA, 0.6)
        AiPhile.textBGoutline(frame, f'Detection: Pyzbar', (30,80), scaling=0.5,text_color=(AiPhile.MAGENTA ))

        # x, x1 = hull_points[0][0], hull_points[1][0]
        # y, y1 = hull_points[0][1], hull_points[1][1]
        qr_detected= True
        stop_code=True
        old_points = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)

        cv.circle(frame, pt1, 3, AiPhile.GREEN, 3)
        cv.circle(frame, pt2, 3, (255, 0, 0), 3)
        cv.circle(frame, pt3, 3, AiPhile.YELLOW, 3)
        cv.circle(frame, pt4, 3, (0, 0, 255), 3)
    if qr_detected and stop_code==False:
        AiPhile.textBGoutline(frame, f'Detection: Optical Flow', (30,80), scaling=0.5,text_color=AiPhile.YELLOW)

        new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_points = new_points 
        new_points=new_points.astype(int)

        frame =AiPhile.fillPolyTrans(frame, new_points, AiPhile.GREEN, 0.6)
        AiPhile.textBGoutline(frame, f'Detection: Optical Flow', (30,80), scaling=0.5,text_color=AiPhile.GREEN)
        cv.circle(frame, (new_points[1]), 3,AiPhile.BLACK, 2)
        cv.circle(frame, (new_points[0]), 3,AiPhile.GREEN, 2)

        x, y = new_points[0].ravel()
        x1, y1 = new_points[1].ravel()  
        # print(x, y )
        qr_height = eucaldainDistance(x, y, x1, y1)
        distance = distanceFinder(focal_length, KNOWN_WIDTH, qr_height)
        AiPhile.textBGoutline(frame, f'Distance: {round(distance,2)} cm', (340,50), scaling=0.8,bg_color=AiPhile.BLACK, text_color=AiPhile.GREEN )

    old_gray = gray_frame.copy()
    # press 'r' to reset the window
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    fps = frame_counter/(time.time()-starting_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("image", frame)


# close all open windows
cv.destroyAllWindows() 
cap.release()