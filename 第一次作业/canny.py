import numpy as np
import cv2
from scipy.ndimage import convolve  # 用于卷积操作


class Canny:

    def __init__(self, img, Gaussian_kernel, Gaussian_sd, low_threshold, high_threshold):
        self.img = img
        self.Gaussian_kernel = Gaussian_kernel
        self.Gaussian_sd = Gaussian_sd
        self.magnitude = None
        self.angle = None
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.nms_result = None

    def Gaussian_ambiguity(self):
        self.img = cv2.GaussianBlur(self.img, (self.Gaussian_kernel, self.Gaussian_kernel), self.Gaussian_sd)
        return self.img

    def get_gradient(self):
        # 转换图像为灰度
        if len(self.img.shape) > 2:  # 如果不是灰度图
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.img

        # 使用cv2.Sobel获取X和Y方向的梯度
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度的幅度和方向
        self.magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        self.angle = np.arctan2(gradient_y, gradient_x)  # 这里保持使用弧度

    def non_maximum_suppression(self):
        # 初始化非极大值抑制结果矩阵
        M, N = self.magnitude.shape
        Z = np.zeros((M, N), dtype=np.float32)

        # 将梯度方向调整到[0,180]范围
        angle = self.angle % 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    # 根据梯度方向，比较当前像素与邻域像素的梯度幅度
                    # 梯度方向为0度
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = self.magnitude[i, j + 1]
                        r = self.magnitude[i, j - 1]
                    # 梯度方向为45度
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = self.magnitude[i + 1, j - 1]
                        r = self.magnitude[i - 1, j + 1]
                    # 梯度方向为90度
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = self.magnitude[i + 1, j]
                        r = self.magnitude[i - 1, j]
                    # 梯度方向为135度
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = self.magnitude[i - 1, j - 1]
                        r = self.magnitude[i + 1, j + 1]

                    # 只保留梯度方向上的局部最大值点
                    if self.magnitude[i, j] >= q and self.magnitude[i, j] >= r:
                        Z[i, j] = self.magnitude[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    # 忽略边缘问题
                    pass
        self.nms_result = Z
        return Z

    def threshold_and_link_edges(self):
        # 应用双阈值检测
        M, N = self.nms_result.shape
        res = np.zeros((M, N), dtype=np.int32)

        # 定义强边缘和弱边缘
        strong_i, strong_j = np.where(self.nms_result >= self.high_threshold)
        weak_i, weak_j = np.where((self.nms_result <= self.high_threshold) & (self.nms_result >= self.low_threshold))

        res[strong_i, strong_j] = 255
        res[weak_i, weak_j] = 75

        # 创建一个标记数组，用于标记已经检查过的弱边缘点
        checked = np.zeros_like(res, dtype=bool)

        # 边缘连接：将弱边缘连接到强边缘
        def link_edges(i, j):
            if res[i, j] == 75 and not checked[i, j]:
                # 标记当前弱边缘点为已检查
                checked[i, j] = True
                # 检查周围8个邻域像素
                for x in range(max(0, i - 1), min(i + 2, M)):
                    for y in range(max(0, j - 1), min(j + 2, N)):
                        # 如果邻域像素中有强边缘，则当前弱边缘点也变为强边缘
                        if res[x, y] == 255:
                            res[i, j] = 255
                            return
                        # 否则，递归地应用相同的逻辑到邻域的弱边缘点
                        elif res[x, y] == 75:
                            link_edges(x, y)

        # 对所有弱边缘点应用边缘连接逻辑
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                link_edges(i, j)

        # 将未连接到强边缘的弱边缘点去除
        for i in range(M):
            for j in range(N):
                if res[i, j] == 75:
                    res[i, j] = 0

        return res

    def execute_canny_detection(self):
        self.Gaussian_ambiguity()
        self.get_gradient()
        self.non_maximum_suppression()
        final_edges = self.threshold_and_link_edges()
        return final_edges.astype(np.uint8)

# if __name__ == '__main__':
#     img = cv2.imread('data/img-3.jpg')
#     canny_detector = Canny(img, 5, 1.4, 13, 15)
#     final_edges = canny_detector.execute_canny_detection()
#     cv2.imshow('Canny Edges', final_edges)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
