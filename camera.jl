using PyCall

cv = pyimport("cv2")
glob = pyimport("glob")
np = pyimport("numpy")

critera = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)