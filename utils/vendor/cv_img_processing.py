# @Author: Dedsecer
# @Sourceurl: https://github.com/DedSecer/AI-Plays-ForzaHorizon4
# rebuilt by APLaS
import cv2
import numpy as np

def extract_blue(screen_in):
    # 避免不必要的完整HSV转换，只提取需要的H通道
    hsv = cv2.cvtColor(screen_in, cv2.COLOR_RGB2HSV)

    # 使用更紧凑的蓝色范围
    mask = cv2.inRange(hsv, np.array([90, 38, 120]), np.array([165, 130, 240]))

    # 使用更高效的方式应用掩码
    # 如果只需要掩码区域而不关心颜色可以优化为:
    # return mask
    # 否则保留原来的方式:
    return cv2.bitwise_and(screen_in, screen_in, mask=mask)

def bird_eye_view(img):
    img_size = (img.shape[1], img.shape[0])

    # 预计算这些值，避免重复计算
    width_half = img.shape[1] * 0.5
    bot_width_half = img.shape[1] * 0.35  # .70/2
    mid_width_half = img.shape[1] * 0.025  # .05/2
    height_pos = img.shape[0] * 0.45
    bottom_pos = img.shape[0] * 0.70

    src = np.float32(
        [
            [width_half - mid_width_half, height_pos],
            [width_half + mid_width_half, height_pos],
            [width_half + bot_width_half, bottom_pos],
            [width_half - bot_width_half, bottom_pos],
        ]
    )

    offset = img_size[0] * 0.25
    dst = np.float32(
        [
            [offset, 0],
            [img_size[0] - offset, 0],
            [img_size[0] - offset, img_size[1]],
            [offset, img_size[1]],
        ]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


def bird_view_processing(screen_in, resize_width=160, resize_height=90):
    processed_image = bird_eye_view(screen_in)
    # cv2.imshow('bird_view', cv2.resize(processed_image, (480, 270)))
    processed_image = extract_blue(processed_image)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.resize(processed_image, (resize_width, resize_height))
    return processed_image

def crop_screen(screen_in, trim_rate=0.3):
    # this function is for cropping the edge part of a screen(in order to get rid of the game UI).
    # screen_in is the opencv screen array
    # trim_rate is how much you want to cut, it a percentage number.
    height, width, _ = screen_in.shape
    padding_w = int(width * trim_rate)
    padding_h = int(height * trim_rate)
    return screen_in[padding_h:, :, :]


def edge_processing(screen_in, resize_width=160, resize_height=90):
    # 一次性调整大小
    screen_resized = cv2.resize(screen_in, (resize_width, resize_height))

    # 直接进行边缘检测
    edges = cv2.Canny(screen_resized, 200, 255)

    # 预定义ROI坐标
    roi_corners = np.array(
        [
            [
                (0, resize_height // 2),
                (resize_width // 2, 0),
                (resize_width - 1, resize_height // 2),
                (resize_width - 1, resize_height - 1),
                (0, resize_height - 1),
            ]
        ],
        dtype=np.int32,
    )

    # 创建蒙版并应用
    mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.fillPoly(mask, roi_corners, 255)

    return cv2.bitwise_and(edges, mask)
