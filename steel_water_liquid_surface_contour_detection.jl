using PyCall
using TyImages
cv2 = pyimport("cv2")
np = pyimport("numpy")

function contour_detected(frame)
    canny_thresholds = (100, 200)  # 可能需要进一步调整
    min_contour_area = 500  # 可能需要进一步调整

    # 将 frame 转换为 NumPy 数组
    frame_array = np.array(frame)
    gray = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)

    # 新增：文字掩膜消除
    _, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # 捕获高亮文字
    gray = cv2.inpaint(gray, text_mask, 3, cv2.INPAINT_TELEA)  # 修复文字区域

    # 新增：热辐射降噪
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blurred, canny_thresholds[1], canny_thresholds[2])
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选轮廓
    valid_contours = []
    for cnt in contours
        area = cv2.contourArea(cnt)
        if area > min_contour_area
            # 计算轮廓的周长
            perimeter = cv2.arcLength(cnt, true)
            # 轮廓近似
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, true)
            # 筛选出接近椭圆形状的轮廓
            if length(approx) > 10  # 椭圆形状通常有较多的顶点
                push!(valid_contours, approx)
            end
        end
    end

    # 绘制轮廓
    result_frame = cv2.drawContours(frame_array, valid_contours, -1, (0, 0, 255), 3)  # 使用红色和更粗的线条

    # 将结果转换回 cv2.UMat
    result_frame_umat = cv2.UMat(result_frame)

    return result_frame_umat
end

function video_process(input_video)
    println("success")
    cap = cv2.VideoCapture(input_video)
    width = Int(cap.get(3))
    height = Int(cap.get(4))
    println(width)
    println(height)
    fps = cap.get(5)
    path = "data/part2/"
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter("$path/output.mp4", fourcc, fps, (width, height))

    fnum = cap.get(7)
    for i in 1:fnum
        ret, frame = cap.read()
        if !ret
            println("Can't read frame (stream end?)")
            break
        end
        frame = contour_detected(frame)
        out.write(frame)
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
