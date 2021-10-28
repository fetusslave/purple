import cv2
import mediapipe as mp
import numpy as np
from math import atan, pi


mpdraw = mp.solutions.drawing_utils
mpholistic = mp.solutions.holistic
holistic = mpholistic.Holistic()

cap = cv2.VideoCapture(0)

height = 720
width = 1280



def getxy(landmarks):
    global width, height
    points = []
    for i in landmarks:
        points.append([int(i.x * width), int(i.y * height)])
    return points

def draw(landmarks):
    global width, height, w, h
    points = getxy(landmarks)

    distance = (((landmarks[9].x-landmarks[10].x)**2)+((landmarks[9].y-landmarks[10].y)**2)+((abs(landmarks[9].z)-abs(landmarks[10].z))**2))**0.5

    b_points = np.array([points[11], points[12], [int(points[24][0]-(distance*2.8)), points[24][1]], [int(points[23][0]+(distance*2.8)), points[23][1]]], np.int32)
    arml_points = np.array([points[11], points[13], points[15]], np.int32)
    armr_points = np.array([points[12], points[14], points[16]], np.int32)
    legl_points = np.array([points[23], points[25], points[27], points[29], points[31]], np.int32)
    legr_points = np.array([points[24], points[26], points[28], points[30], points[32]], np.int32)


    r = int(2857*distance)
    cv2.fillPoly(w, pts=[b_points], color=(255, 0, 255))
    cv2.circle(w, tuple(points[0]), r, (255, 0, 255), -1)
    cv2.circle(w, tuple(points[2]), int(r/5), (0, 0, 0), -1)
    cv2.circle(w, tuple(points[5]), int(r/5), (0, 0, 0), -1)
    cv2.circle(w, (int(points[2][0] + (r / 10)), int(points[2][1] + (r / 10))), int(r / 20), (255, 255, 255), -1)
    cv2.circle(w, (int(points[5][0] + (r / 10)), int(points[5][1] + (r / 10))), int(r / 20), (255, 255, 255), -1)
    #cv2.line(w, tuple(points[9]), tuple(points[10]), (0, 0, 0), 10)
    cv2.polylines(w, [arml_points], False, (0, 0, 0), 10)
    cv2.polylines(w, [armr_points], False, (0, 0, 0), 10)
    cv2.polylines(w, [legl_points], False, (0, 0, 0), 10)
    cv2.polylines(w, [legr_points], False, (0, 0, 0), 10)

def drawhand(landmarks):
    points = getxy(landmarks)
    finger = []
    for i in range(2, 21):
        finger.append(points[i])
        if i%4 == 0:
            finger = np.array(finger, np.int32)
            cv2.polylines(w, [finger], False, (0, 0, 0), 10)
            finger = []


def drawmouth(landmarks):
    points = getxy(landmarks)
    #lip_u_o = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    #lip_l_o = [375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
    #lip_u_i = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    #lip_l_i = [324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    lip_u = points[13]
    lip_l = points[14]
    lip_left = points[308]
    lip_right = points[78]
    center = (int((lip_left[0]+lip_right[0])/2), int((lip_left[1]+lip_right[1])/2))
    x_axis_length = int(((((lip_left[0]-lip_right[0])**2)+((lip_left[1]-lip_right[1])**2))**0.5)/2)
    y_axis_length = int(((((lip_u[0] - lip_l[0]) ** 2) + ((lip_u[1] - lip_l[1]) ** 2)) ** 0.5)/2)
    #print(x_axis_length, y_axis_length)
    if y_axis_length < 5:
        y_axis_length = 5
    angle = (atan(y_axis_length/x_axis_length)*180)/pi
    if lip_left[1] < lip_right[1]:
        angle *= -1
    cv2.ellipse(w, center, (x_axis_length, y_axis_length), angle, 0, 360, (0, 0, 0), -1)



while cap.isOpened():
    s, img = cap.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = holistic.process(image)

    w = np.ones((720, 1280, 3))

    if res.pose_landmarks:
        draw(res.pose_landmarks.landmark)
    if res.left_hand_landmarks:
        drawhand(res.left_hand_landmarks.landmark)
    if res.right_hand_landmarks:
        drawhand(res.right_hand_landmarks.landmark)

    mpdraw.draw_landmarks(img, res.pose_landmarks, mpholistic.POSE_CONNECTIONS)

    if res.face_landmarks:
        drawmouth(res.face_landmarks.landmark)


    cv2.imshow('', w)

    cv2.waitKey(1)
