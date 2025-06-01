from .capture_processing import ImgHelper
from .vendor import cv_img_processing as cv_img_processing
from .vendor import grabscreen
import cv2
import numpy as np


def contrast_modern_gray(gray: np.ndarray, contrast_value: float) -> np.ndarray:
    """
    Modern 模式：对单通道灰度图做对比度调整（相当于 Photoshop 默认的 Brightness/Contrast）。
    参数:
      - gray:           np.ndarray，单通道灰度图（uint8，范围 [0,255]）
      - contrast_value: 对比度滑块，范围 [-100, +100]
    返回:
      - dst:            对比度调整后的灰度图（uint8，范围 [0,255]）
    """
    # 1. 将滑块值归一化到 [-1, +1]
    c = float(contrast_value) / 100.0

    # 2. 计算缩放因子 F
    if c >= 0:
        F = 1.0 / (1.0 - c)  # c 越接近 1，F 越大
    else:
        F = 1.0 + c  # c < 0 时，F 线性缩小到 [0,1)

    # 3. 归一化灰度到 [0,1]
    gray_norm = gray.astype(np.float32) / 255.0

    # 4. 拉伸/压缩：围绕 0.5 进行缩放
    dst_norm = (gray_norm - 0.5) * F + 0.5

    # 5. 裁剪到 [0,1]
    dst_norm = np.clip(dst_norm, 0.0, 1.0)

    # 6. 还原到 [0,255] 并转 uint8
    dst = (dst_norm * 255.0).round().astype(np.uint8)
    return dst


def build_modern_lut(contrast_value: float) -> np.ndarray:
    """
    为 Modern 模式生成 256×1 的查找表（LUT），
    输入像素 0~255，输出对比度增强后像素 0~255。
    """
    c = float(contrast_value) / 100.0
    if c >= 0:
        F = 1.0 / (1.0 - c)
    else:
        F = 1.0 + c

    lut = np.zeros((256,), dtype=np.uint8)
    for p in range(256):
        p_norm = p / 255.0
        new_norm = (p_norm - 0.5) * F + 0.5
        new_norm = min(max(new_norm, 0.0), 1.0)
        lut[p] = int(round(new_norm * 255.0))
    return lut


global_lut = build_modern_lut(50)


def contrast_modern_ycrcb_lut(bgr_img: np.ndarray, contrast_value: float) -> np.ndarray:
    """
    在 YCrCb 色彩空间下，仅对 Y 通道做 Modern 对比度映射（使用 LUT 加速），
    最后再合并回 BGR。速度相比 Lab 方案能提升约 20%~30%。

    输入：
      - bgr_img:        uint8，BGR 彩色图（H×W×3）
      - contrast_value: 对比度滑块，范围 [-100, +100]
    输出：
      - enhanced_bgr:   对比度增强后的 uint8 BGR 图（H×W×3）
    """
    # 1. 把 BGR 转到 YCrCb
    #    Y 通道是亮度，Cr、Cb 保留色度。
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 2. 生成一次 LUT，并对 Y 通道做查表映射
    lut = build_modern_lut(contrast_value)
    y_enhanced = cv2.LUT(y, lut)

    # 3. 把增强后的 Y 通道与原 Cr、Cb 合并
    ycrcb_enhanced = cv2.merge((y_enhanced, cr, cb))

    # 4. 再从 YCrCb 转回 BGR，得到最终彩色图
    enhanced_bgr = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)
    return enhanced_bgr


def cutimg(img, x=480, y=135, w=960, h=640):
    """
    裁剪图像到指定区域。
    参数:
      - img: 输入图像（np.ndarray）
      - x, y: 裁剪区域左上角坐标
      - w, h: 裁剪区域宽度和高度
    返回:
      - cropped_img: 裁剪后的图像（np.ndarray）
    """
    return img[y : y + h, x : x + w]


def get_adjusted_view(frame):
    """
    获取调整后的视图，裁剪并应用对比度增强。
    参数:
      - frame: 输入图像（np.ndarray）
    返回:
      - adjusted_view: 调整后的视图（np.ndarray）
    """
    cropped_frame = cutimg(frame, 480, 135, 960, 640)
    adjusted_view = contrast_modern_ycrcb_lut(cropped_frame, 50)
    return adjusted_view

class CaptureGuideline:
    def __init__(self):
        self.grabber = grabscreen.ScreenGrabber()

    def get_digit(self, frame):
        return ImgHelper.capture_digit_region(frame)

    def get_bule_bird_eye_view(self, frame):
        bird_eye_view = cv_img_processing.bird_eye_view(frame)
        resized_bird_eye_view = cv2.resize(bird_eye_view, (240, 136))
        blue_img = cv_img_processing.extract_blue(
            cv2.cvtColor(resized_bird_eye_view, cv2.COLOR_BGR2RGB)
        )
        return blue_img

    def get_currunt_key_region(self):
        frame = self.grabber.grab()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        digit_region = self.get_digit(frame)
        _adjusted_view = get_adjusted_view(frame)
        return digit_region, _adjusted_view

    def get_frame(self):
        frame = self.grabber.grab()
        return frame
