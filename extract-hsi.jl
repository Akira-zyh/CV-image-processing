using PyCall
using TyImages
cv2 = pyimport("cv2")
image = cv2.imread("data/testImage/straight_lines1.jpg")

function bgr2hsi(img_bgr)
    img_float = float.(img_bgr) / 255.0
    r, g, b = img_float[:, :, 1], img_float[:, :, 2], img_float[:, :, 3]

    # Intensity计算
    I = (r + g + b) / 3.0

    # Saturation计算
    min_rgb = min.(r, g, b)
    S = 1.0 .- (3.0 ./ (r + g + b .+ 1e-6)) .* min_rgb

    # Hue计算
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = sqrt.((r - g) .^ 2 + (r - b) .* (g - b))
    H = acos.(numerator ./ (denominator .+ 1e-6))
    H[b.>g] = 2π .- H[b.>g]
    H = rad2deg.(H)

    return cat(H, S, I, dims=3)
end

hsi = bgr2hsi(image)  # 替换原HLS转换

sThrld = (170, 255)
sblThrld = (40, 200)
iThrld = 0.5

# 颜色空间转换
hsi = bgr2hsi(image)

# 通道提取
iChannel = hsi[:, :, 3]
sChannel = hsi[:, :, 2]

# 颜色过滤
yellow_mask = (hsi[:, :, 1] .> 50) .& (hsi[:, :, 1] .< 70) .& (hsi[:, :, 2] .> 0.5)
white_mask = (hsi[:, :, 2] .< 0.2) .& (hsi[:, :, 3] .> 0.8)

# 边缘检测
sobelx = cv2.Sobel(iChannel, cv2.CV_64F, 1, 0)
abs_sobelx = abs.(sobelx)
scaled_sobel = UInt8.(round.(255 * abs_sobelx ./ maximum(abs_sobelx)))

# 阈值处理
sblBin = (scaled_sobel .>= sblThrld[1]) .& (scaled_sobel .<= sblThrld[2])
iBin = iChannel .> iThrld

# 结果合并
resultBin = (yellow_mask .| white_mask .| sblBin) .& iBin
imshow(resultBin)