import numpy as np


class HoughCircleDetector:
    def __init__(self, min_radius, max_radius, threshold, nms_radius):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.threshold = threshold
        self.nms_radius = nms_radius  # NMS的局部范围半径

    def detect_circles(self, edge_image):
        height, width = edge_image.shape
        accumulator = np.zeros((height, width, self.max_radius - self.min_radius))

        # 为每个边缘点投票
        for x in range(width):
            for y in range(height):
                if edge_image[y, x] == 255:  # 边缘点
                    for r in range(self.min_radius, self.max_radius):
                        for theta in range(0, 360, 10):  # 使用更小的角度步长
                            a = int(x - r * np.cos(np.deg2rad(theta)))
                            b = int(y - r * np.sin(np.deg2rad(theta)))
                            if 0 <= a < width and 0 <= b < height:
                                accumulator[b, a, r - self.min_radius] += 1

        # 应用非最大值抑制(NMS)
        return self.apply_nms(accumulator)

    def apply_nms(self, accumulator):
        circles = []
        height, width, _ = accumulator.shape
        for r in range(self.min_radius, self.max_radius):
            for x in range(width):
                for y in range(height):
                    if accumulator[y, x, r - self.min_radius] > self.threshold:
                        if self.is_local_max(accumulator, x, y, r - self.min_radius):
                            circles.append((x, y, r))

        # 过滤相近的圆
        filtered_circles = []
        for circle in circles:
            x, y, r = circle
            if all((x - cx) ** 2 + (y - cy) ** 2 > self.nms_radius ** 2 for cx, cy, cr in filtered_circles):
                filtered_circles.append(circle)

        return filtered_circles

    def is_local_max(self, accumulator, x, y, r):
        for dx in range(-3, 4):  # 检查更大范围内的局部最大值
            for dy in range(-3, 4):
                nx, ny = x + dx, y + dy
                if nx >= 0 and nx < accumulator.shape[1] and ny >= 0 and ny < accumulator.shape[0]:
                    if accumulator[ny, nx, r] > accumulator[y, x, r]:
                        return False
        return True
