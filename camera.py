import numpy as np
import cv2 as cv
import glob
import os

# ============================= Some Variables ================================

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# ============================= Folder Path ================================

original_path = "data/part3/original/"
calibresult_path = "data/part3/calibResult/"
marked_path = "data/part3/markedOriginal/"
camera_params_path = "data/part3/cameraParams/"
ReconResult_path = "data/part3/3dConstruction/"
images = glob.glob(os.path.join(original_path, '*.jpg'))

# ============================= Corner Marking ================================

# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (6,8), None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)

#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)

#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (6,8), corners2, ret)
#         cv.imshow('img', img)
#         save_fname = "marked_" + fname
#         cv.imwrite(fname, img)
#         cv.waitKey()

# ============================= Distortion ================================

# for fname in images:
#     img = cv.imread(fname)
#     cv.waitKey()
#     gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#     cv.imshow('preview', gray)
#     ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#     h, w = img.shape[:2]
#     newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#     # undistort
#     dst = cv.undistort(img, mtx, dist, None, newcameramtx)
#     np.savez(f"camera_params_{fname[::-4]}", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    
#     # crop the image
#     x, y, w, h = roi
#     dst = dst[y:y+h, x:x+w]
#     cv.imwrite(f'calibresult_{fname}', dst)

# ============================= 3D Reconstruction ================================

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

for i in range(10):
    with np.load(os.path.join(camera_params_path, f"camera_params_g{i}.npz")) as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    fname = f"{i}.jpg"

    img = cv.imread(os.path.join(original_path, fname))
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (6, 8), None)

    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        imgts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgts)
        cv.imshow('img', img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(os.path.join(ReconResult_path, fname[:6] + '.png'), img)
cv.destroyAllWindows()