import cv2
from canny import Canny
from hough import HoughCircleDetector


def draw(img, circles):
    # 在原图上绘制检测到的圆
    for (x, y, r) in circles:
        # 绘制圆周
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        # 绘制圆心
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

    # 保存结果图像
    cv2.imwrite('data/detected_circles.jpg', img)


img = cv2.imread('data/img-5.jpg')
canny_detector = Canny(img, 5, 1.4, 30, 70)
final_edges = canny_detector.execute_canny_detection()

# 保存Canny处理后的图像
cv2.imwrite('data/final_edges.jpg', final_edges)

hough = HoughCircleDetector(20, 150, 10, 80)
# hough = HoughCircleDetector(20, 30, 10, 50)
circles = hough.detect_circles(final_edges)
print(circles)
draw(img, circles)
