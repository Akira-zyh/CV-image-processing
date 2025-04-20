using PyCall # A library which allows Julia call Python's library
using TyImages # A library for processing image
using Glob
cv = pyimport("cv2")

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = zeros(Float32, 6*7, 3)
x = collect(0:6)
y = collect(0:5)
grid_x, grid_y = meshgrid(x, y)
# objp[:, 1:2] = permutedims(reshape(collect(0:6*7-1), 7, 6), (2, 1))
objp[:, 1] = vec(grid_x)
objp[:, 2] = vec(grid_y)

objpoints = []
imgpoints = []

for i in 0:15
    img = cv.imread("data/part3/$i.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), nothing)

    if ret
        push!(objpoints, objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11),  (-1, -1), criteria)
        push!(imgpoints, corners2)
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)
    end
end

cv.destroyAllWindows()