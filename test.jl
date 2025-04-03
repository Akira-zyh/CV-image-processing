using PyCall
using TyImages
cv2 = pyimport("cv2")
np = pyimport("numpy")

function contour_detected(frame)
    canny_thresholds = (30, 150)
    min_contour_area = 1000

    # 创建可写副本
    drawing_frame = np.copy(frame)

    # 预处理流程
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, canny_thresholds[1], canny_thresholds[2])

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # 内存布局修正
    if !drawing_frame.flags["C_CONTIGUOUS"]
        drawing_frame = np.ascontiguousarray(drawing_frame)
    end

    # 颜色空间转换（解决UMat兼容性）[8]
    drawing_frame = cv2.cvtColor(drawing_frame, cv2.COLOR_BGR2RGB)
    cv2.drawContours(drawing_frame, valid_contours, -1, (0, 255, 0), 2)
    result_frame = cv2.cvtColor(drawing_frame, cv2.COLOR_RGB2BGR)

    return result_frame
end

function video_process(input_video)
    cap = cv2.VideoCapture(input_video)
    width = Int(cap.get(3))
    height = Int(cap.get(4))
    fps = cap.get(5)
    
    # 修正编码器参数[1]
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter("data/part2/output2.mp4", fourcc, fps, (width, height))

    total_frames = Int(cap.get(7))
    for i in 1:total_frames
        ret, frame = cap.read()
        !ret && break
        
        processed_frame = contour_detected(frame)
        out.write(processed_frame)
        
        # 进度显示优化
        if i % 50 == 0
            progress = round(i / total_frames * 100, digits=1)
            @info "Progress: $progress% ($i/$total_frames)"
        end
    end
    
    # 资源释放
    cap.release()
    out.release()
    cv2.destroyAllWindows()
end

# 执行处理
input_video = "data/part2/Experiment2.mp4"
video_process(input_video)