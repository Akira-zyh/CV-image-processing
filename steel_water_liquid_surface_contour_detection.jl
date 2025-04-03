using PyCall
using TyImages
cv2 = pyimport("cv2")
np = pyimport("numpy")

function contour_detected(frame)
    canny_thresholds = (130, 180)
    min_contour_area = 450

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    result_frame = cv2.drawContours(frame, valid_contours, -1, (0, 255, 0), 2)

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