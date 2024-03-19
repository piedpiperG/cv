import cv2
import numpy as np

# 读取图片
img = cv2.imread('data/img-3.jpg')  # 将'your_image.jpg'替换为你的图片文件名
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊，减少图像噪声
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# 保存Canny边缘检测的结果
cv2.imwrite('canny_edges.jpg', edges)

# 使用霍夫变换检测圆形
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                           param1=50, param2=30, minRadius=20, maxRadius=100)

# 绘制检测到的圆形
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 绘制圆心
        cv2.circle(img, (i[0], i[1]), 1, (0, 100, 100), 3)
        # 绘制圆轮廓
        cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 255), 2)

# 保存霍夫圆检测的结果
cv2.imwrite('hough_circles.jpg', img)

# 如果你需要显示图片，可以解除以下代码的注释
# cv2.imshow('Detected Circles', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
