using PyCall
using TyImages

cv2 = pyimport("cv2")
img = cv2.imread("data/testImage/test1.jpg")
rimg = reverse(img, dims=3)
imshow(rimg)