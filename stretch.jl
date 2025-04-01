using PyCall
using TyImages
cv2 = pyimport("cv2")
image = cv2.imread("data/testImage/test1.jpg")

sThrld = (170, 255)
sblThrld = (40, 200)

hls = float(cv2.cvtColor(image, cv2.COLOR_BGR2HLS))
lChannel = hls[:, :, 2]
sChannel = hls[:, :, 3]

sBin = (sChannel .> sThrld[1]) .& (sChannel .<= sThrld[2])
imshow(sBin)