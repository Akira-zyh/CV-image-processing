using PyCall
using TyImages
cv2 = pyimport("cv2")
np = pyimport("numpy")

function contour_detected(frame)
    canny_thresholds = (40, 60)
    min_contour_area = 850
    max_contour_area = 100000

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 3, 5)

    # 形态学操作改善二值图像
    kernel_close = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # 热辐射降噪
    processed = cv2.bilateralFilter(binary, 9, 75, 75)

    blurred = cv2.GaussianBlur(processed, (7, 7), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, canny_thresholds[1], canny_thresholds[2])

    # 边缘膨胀，使用更大的核以连接断开的边缘
    kernel = np.ones((5, 5), np.uint8)  # 增加核大小
    edges = cv2.dilate(edges, kernel, iterations=2)  # 增加迭代次数

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选面积足够大的轮廓
    # valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    valid_contours = [cnt for cnt in contours if
                      (cv2.contourArea(cnt) > min_contour_area &&
                       cv2.contourArea(cnt) < max_contour_area)]

    result_frame = frame  # 不创建副本，直接使用原始帧

    if !isempty(valid_contours)
        # 找到最大的轮廓
        largest_contour_idx = argmax([cv2.contourArea(cnt) for cnt in valid_contours])
        largest_contour = valid_contours[largest_contour_idx]

        contour_array = largest_contour.get()
        contour_size = PyCall.pybuiltin("len")(contour_array)

        # 使用椭圆拟合替代多边形近似，实现平滑轮廓
        if contour_size >= 5  # 椭圆拟合至少需要5个点
            try
                # 椭圆拟合
                ellipse = cv2.fitEllipse(largest_contour)
                # 绘制椭圆 - 红色，宽度2
                result_frame = cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
            catch e
                # 如果椭圆拟合失败，回退到多边形近似
                println("椭圆拟合失败: $e")
                epsilon = 0.02 * cv2.arcLength(largest_contour, true)
                approx = cv2.approxPolyDP(largest_contour, epsilon, true)
                result_frame = cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
            end
        else
            # 如果点不够多，回退到多边形近似
            epsilon = 0.02 * cv2.arcLength(largest_contour, true)
            approx = cv2.approxPolyDP(largest_contour, epsilon, true)
            result_frame = cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
        end
    end

    return result_frame
end

function video_process(input_video)
    println("succees")
    cap = cv2.VideoCapture(input_video)
    width = Int(cap.get(3))
    height = Int(cap.get(4))
    println(width)
    println(height)
    fps = cap.get(5)
    path = "data/part2/"
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter("data/part2/output.mp4", fourcc, fps, (width, height))

    fnum = cap.get(7)
    for i in 1:fnum
        ret, frame = cap.read()
        if !ret
            println("Can't read frame (stream end?)")
            break
        end
        frame = cv2.UMat(frame)
        out.write(contour_detected(frame))
        if i % 50 == 0
            progress = round(i / fnum * 100, digits=1)
            @info "Progress: $progress% ($i/$fnum)"
        end
    end
    cap.release()
    out.release()
end

input_video = "data/part2/Experiment2.mp4"
video_process(input_video)
