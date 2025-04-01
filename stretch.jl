using PyCall
using TyImages
cv2 = pyimport("cv2")
image = cv2.imread("data/testImage/straight_lines1.jpg")

sThrld = (170, 255)
sblThrld = (195, 200)

hls = float(cv2.cvtColor(image, cv2.COLOR_BGR2HLS))
lChannel = hls[:, :, 2]
sChannel = hls[:, :, 3]

sBin = (sChannel .> sThrld[1]) .& (sChannel .<= sThrld[2])

sobelx = cv2.Sobel(lChannel, cv2.CV_64F, 1, 0)
abs_sobelx = abs.(sobelx)
normSobelx = 255 * abs_sobelx ./maximum(abs_sobelx)

normSobelx = round.(normSobelx)
scaled_sobel = convert.(UInt8, normSobelx)
imshow(scaled_sobel)

hSblBin = (scaled_sobel .>= sblThrld[1]) .& (scaled_sobel .<= sblThrld[2])
lBin = lChannel .> 100

resultBin = (hSblBin .| sBin) .& lBin
imshow(resultBin)