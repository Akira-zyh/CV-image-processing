using PyCall

# 配置Python环境（需提前执行 pip install opencv-python numpy）
cv2 = pyimport("cv2")
np = pyimport("numpy")
os = pyimport("os")

function process_video(input_path::String, output_dir::String)
    # 初始化资源对象（关键修复1：作用域提升）
    cap = nothing
    out = nothing
    
    try
        # 创建输出目录（关键修复2：路径处理）
        os.makedirs(output_dir, exist_ok=true)
        output_path = joinpath(output_dir, "processed_video.mp4")

        # 视频读取初始化（参考网页1[1]   ）
        @info "正在打开视频文件：" * abspath(input_path)
        cap = cv2.VideoCapture(input_path)
        !cap.isOpened() && error("视频打开失败，请检查路径和文件格式")

        # 获取视频参数（参考网页3[3]   ）
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = Int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = Int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = Int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 视频写入器初始化（关键修复3：编解码器兼容性[6]   ）
        fourcc = cv2.VideoWriter_fourcc("mp4v")  # 使用字符串参数替代字符序列
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        !out.isOpened() && error("视频写入器初始化失败，尝试更换编码器为'avc1'")

        # 处理参数配置（参考网页2[2]   ）
        canny_thresholds = (30, 150)
        min_contour_area = 1000

        frame_count = 0
        @info "开始处理视频: $input_path"
        @info "参数: 分辨率($width×$height) FPS=$fps 总帧数=$total_frames"

        while cap.isOpened()
            ret, frame = cap.read()
            !ret && break  # 视频结束则退出

            # 创建可写副本（关键修复4：内存连续性[4]   ）
            drawing_frame = np.copy(frame)  # 创建独立内存空间

            # 图像处理流程（参考网页4[4]   ）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15,15), 0)
            edges = cv2.Canny(blurred, canny_thresholds[1]   , canny_thresholds[2]   )

            # 轮廓检测（参考网页3[3]   ）
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            # 内存布局强制连续（关键修复5：UMat兼容性[8]   ）
            if !drawing_frame.flags["C_CONTIGUOUS"]
                drawing_frame = np.ascontiguousarray(drawing_frame)
            end

            # 绘制轮廓到副本（关键修复6：颜色空间转换[4]   ）
            drawing_frame = cv2.cvtColor(drawing_frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间
            cv2.drawContours(drawing_frame, valid_contours, -1, (0, 255, 0), 2)
            result_frame = cv2.cvtColor(drawing_frame, cv2.COLOR_RGB2BGR)  # 转换回BGR格式

            # 写入处理帧（参考网页6[6]   ）
            out.write(result_frame)
            frame_count += 1

            # 进度显示（每10%显示一次）
            if frame_count % Int(round(total_frames/10)) == 0
                progress = round(frame_count / total_frames * 100, digits=1)
                @info "处理进度: $progress% ($frame_count/$total_frames)"
            end

            # 内存管理（参考网页7[7]   ）
            if frame_count % 100 == 0
                cv2.waitKey(1)  # 允许OpenCV处理内部事件
                GC.gc()         # Julia垃圾回收
            end
        end

        # 资源释放（关键修复7：双重检查机制[1]   ）
        cap.isOpened() && cap.release()
        out.isOpened() && out.release()
        cv2.destroyAllWindows()

        @info "处理完成！输出文件已保存至: $(abspath(output_path))"
        return output_path

    catch e
        # 增强错误处理（参考网页8[8]   ）
        @error "处理失败：" exception=(e, catch_backtrace())
        
        # 安全释放资源（关键修复8：异常处理）
        try
            cap.isOpened() && cap.release()
            out.isOpened() && out.release()
        catch
            @warn "资源释放过程中发生二次错误"
        end
        cv2.destroyAllWindows()
        
        return nothing
    end
end

# 调用示例（路径需根据实际调整）
input_video = "data/part2/Experiment2.mp4" 
output_directory = "data/part2/processed/"

# 执行处理
result = process_video(input_video, output_directory)