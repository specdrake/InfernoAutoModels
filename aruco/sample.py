import cv2
import numpy
import cv2.aruco as aruco

ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)

board = aruco.GridBoard_create(
    markersX=2,
    markersY=2,
    markerLength=0.09,
    markerSeparation=0.01,
    dictionary=ARUCO_DICT
)

rvecs, tvecs = None, None

cam = cv2.VideoCapture('gridboardtest.mp4')
while(cam.isOpened()):
    ret, QueryImg = cam.read()
    if ret:
        gray = cv2.cvtColor(QueryImg,cv2.COLOR_BGR2GRAY)
        corners, ids, rejecetedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        if ids is not None and len(ids) == 5:
            for i, corner in zip(ids, corner):
                print('ID: {}; Corners: {}'.format(i, corner))
            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0,0,255))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.imshow('QueryImg', QueryImg)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()