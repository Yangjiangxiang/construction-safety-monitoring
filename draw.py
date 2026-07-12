import cv2
import numpy as np

class ImageDrawer:
    def __init__(self):
        # self.points = [[0, 241], [100, 241], [160, 310], [0, 355]]
        self.points = [[1533, 541], [1260, 500], [1260, 792], [1533 , 792 ]]

    def draw_polygon(self, image):
        w,h,c = image.shape
        # print(w,h,c) 792 1912 3
        points = np.array(self.points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        return image

    def draw_bounding_box(self, image, xmin, ymin, xmax, ymax):
        # 判断四边形内是否有任意两个顶点在矩形框内
        points_inside = [point for point in self.points if xmin <= point[0] <= xmax and ymin <= point[1] <= ymax]
        if len(points_inside) >= 2:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(image, 'Warning', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image

    def is_inside_polygon(self, x, y):
        # 判断点(x, y)是否在四边形内部
        return cv2.pointPolygonTest(np.array(self.points, np.int32), (x, y), False) >= 0

    def show_image(self, image):
        cv2.imshow('Image with Annotations',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 创建ImageDrawer对象并使用示例
    drawer = ImageDrawer()
    img = cv2.imread("test.png")
    img = drawer.draw_polygon(img)
    img = drawer.draw_bounding_box(img, 50, 300, 80, 350)  # 示例输入xmin, ymin, xmax, ymax
    drawer.show_image(img)
