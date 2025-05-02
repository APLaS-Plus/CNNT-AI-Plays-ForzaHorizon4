import PIL.Image
import cv2
import PIL
import numpy as np
import time
from PIL import ImageGrab


class ImgHelper:
    """
    基于OpenCV的截屏功能类，用于获取和处理屏幕截图。
    该类不包含初始化函数。
    """

    @staticmethod
    def capture_screen():
        """
        捕获整个屏幕的截图
        :return: numpy数组格式的截图
        """
        screenshot = ImageGrab.grab()
        screenshot = np.array(screenshot)
        # 将BGR格式转换为RGB格式
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        return screenshot

    @staticmethod
    def capture_screen_region(x, y, width, height):
        """
        捕获屏幕的特定区域
        :param x: 区域左上角的x坐标
        :param y: 区域左上角的y坐标
        :param width: 区域宽度
        :param height: 区域高度
        :return: numpy数组格式的截图
        """
        screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        return screenshot


    @staticmethod
    def capture_region(image, x, y, width, height):
        """
        使用cv2.getRectSubPix从图像中裁剪特定区域
        :param image: 输入的图像（numpy数组）
        :param x: 区域左上角的x坐标
        :param y: 区域左上角的y坐标
        :param width: 区域宽度
        :param height: 区域高度
        :return: 裁剪后的图像区域
        """
        # 确保坐标和尺寸有效
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))

        # 计算中心点
        center_x = x + width / 2
        center_y = y + height / 2

        # 使用cv2.getRectSubPix获取子图像
        return cv2.getRectSubPix(image, (int(width), int(height)), (center_x, center_y))

    @staticmethod
    def save_screenshot(image, filename):
        """
        保存截图到文件
        :param image: numpy数组格式的截图
        :param filename: 保存的文件名
        """
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @staticmethod
    def convert_to_grayscale(image):
        """
        将图像转换为灰度
        :param image: 彩色图像
        :return: 灰度图像
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    

    @staticmethod
    def draw_rectangle(image, x, y, width, height, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制矩形
        :param image: 要绘制的图像
        :param x: 左上角x坐标
        :param y: 左上角y坐标
        :param width: 宽度
        :param height: 高度
        :param color: 颜色，默认为绿色
        :param thickness: 线条粗细
        :return: 绘制了矩形的图像
        """
        result = image.copy()
        cv2.rectangle(result, (x, y), (x + width, y + height), color, thickness)
        return result

    @staticmethod
    def show_image(image):
        image = PIL.Image.fromarray(image)
        image.show()

    @staticmethod
    def warp_perspective(image, points, target_width, target_height):
        """
        将任意四边形的图片区域拉伸成指定大小的矩形

        :param image: 输入图像
        :param points: 原图中四边形的四个顶点坐标，格式为[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                      顶点顺序应为：左上、右上、右下、左下
        :param target_width: 目标矩形的宽度
        :param target_height: 目标矩形的高度
        :return: 拉伸后的矩形图像
        """
        # 确保点是按照左上、右上、右下、左下的顺序排列
        src = np.float32(points)

        # 定义目标矩形的四个顶点坐标
        dst = np.float32(
            [
                [0, 0],  # 左上
                [target_width - 1, 0],  # 右上
                [target_width - 1, target_height - 1],  # 右下
                [0, target_height - 1],  # 左下
            ]
        )

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src, dst)

        # 应用透视变换
        result = cv2.warpPerspective(image, matrix, (target_width, target_height))

        return result

    def capture_digit_region(img):
        """
        Capture the digit region from the image.
        The region is defined as a rectangle with fixed coordinates.
        """
        # Define the coordinates for the digit region (x, y, width, height)
        x, y, w, h = 1710, 912, 160, 96
        # Crop the image to get the digit region
        digit_region = img[y:y+h, x:x+w]
        return digit_region
