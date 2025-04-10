import cv2
import glob

# 棋盘格尺寸
chessboard_size = (8, 6)

# 遍历所有图片
for img_path in glob.glob("GOPR*.jpg"):
    # 读取图像
    img = cv2.imread(img_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 尝试查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到了棋盘格角点
    if ret == True:
        # 亚像素级别角点检测
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))

        # 在图像上绘制角点
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

        # 保存标定后的图像
        cv2.imwrite("marked_" + img_path, img)
