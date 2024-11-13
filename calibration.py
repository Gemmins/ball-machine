import time
import numpy as np
import cv2 as cv
import os

NUM_FRAMES = 200

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] * (56 * NUM_FRAMES)  # 3d point in real world space
imgpoints = [] * (56 * NUM_FRAMES)  # 2d points in image plane.

name = 'cam3'

cam = cv.VideoCapture(0)
i = 0
j = 0
while True:

    img = cam.read()[1]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8, 7), None)

    # If found, add object points, image points (after refining them)
    # Only check every 10th frame
    if ret and j % 10 == 0:

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)
        objpoints.append(objp)

        i += 1

        # Draw and display the corners
        cv.drawChessboardCorners(img, (8, 7), corners2, ret)

    cv.imshow('img', img)

    j += 1

    if i > NUM_FRAMES:  # after n chessboard frames have been captured
        break

    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()

# flag use lu speeds up the calibration massively - not sure how much it affects accuracy tho
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,
                                                  None, flags=cv.CALIB_USE_LU)

print("finished calibrating camera")

img = cv.imread(f'images/{name}.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

np.savez(f"matrices/{name}", mapx=mapx, mapy=mapy)

dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite(f'output/{name}_calibresult.png', dst)
