# CNNT-AI-Plays-ForzaHorizon4

这是一个基于计算机视觉和深度学习的自动驾驶系统，用于在《极限竞速：地平线4》(Forza Horizon 4) 游戏中实现自动驾驶功能。该项目使用神经网络模型从游戏屏幕中识别道路边界、车速等信息，并通过虚拟控制器输出相应的驾驶指令。

## 目录

- [CNNT-AI-Plays-ForzaHorizon4](#cnnt-ai-plays-forzahorizon4)
  - [目录](#目录)
  - [项目概述](#项目概述)
  - [环境配置](#环境配置)
    - [系统需求](#系统需求)
    - [克隆仓库](#克隆仓库)
    - [配置 Python 环境](#配置-python-环境)
      - [方法一：使用 Conda](#方法一使用-conda)
      - [方法二：使用 uv (更快的 pip 替代品)](#方法二使用-uv-更快的-pip-替代品)
    - [目录结构](#目录结构)
  - [数据采集](#数据采集)
    - [屏幕捕获数据](#屏幕捕获数据)
    - [控制器输入监控](#控制器输入监控)
  - [模型训练](#模型训练)
    - [数字识别模型训练](#数字识别模型训练)
    - [模型示例](#模型示例)
  - [运行自动驾驶](#运行自动驾驶)
  - [致谢](#致谢)

## 项目概述

本项目实现了一个基于CNNT（自定义神经网络变换器）的AI系统，可以自动驾驶《极限竞速：地平线4》中的车辆。主要功能包括：

- 实时屏幕捕获和处理
- 车速数字识别（通过自定义轻量级数字检测器）
- 道路边界识别（通过鸟瞰图变换和蓝色提取）
- 驾驶控制输出（方向、加速和刹车）
- 支持键盘和控制器输入监控（用于数据收集）

## 环境配置

### 系统需求

- Windows 10/11 操作系统
- Python 3.10
- CUDA支持的NVIDIA显卡（推荐用于训练和实时推理）
- 安装了《极限竞速：地平线4》游戏

### 克隆仓库

首先，将仓库克隆到本地：
```bash
git clone https://github.com/APLaS-Plus/CNNT-AI-Plays-ForzaHorizon4.git
cd CNNT-AI-Plays-ForzaHorizon4
```

### 配置 Python 环境

#### 方法一：使用 Conda

1. 首先，下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/products/individual)

2. 创建一个新的 conda 环境：
```bash
conda create -n forza-ai python=3.10
conda activate forza-ai
```

3. 安装项目依赖：
```bash
conda install numpy opencv pillow matplotlib
conda install pytorch torchvision -c pytorch
pip install vgamepad pywin32
```

#### 方法二：使用 uv (更快的 pip 替代品)

1. 安装 uv：
```bash
pip install uv
```

2. 创建虚拟环境并安装依赖：
```bash
uv venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 目录结构

```
CNNT-AI-Plays-ForzaHorizon4/
├── utils/                  # 工具函数和类
│   ├── vendor/             # 第三方工具
│   │   ├── grabscreen.py   # 屏幕捕获
│   │   ├── cv_img_processing.py  # 图像处理
│   ├── model/              # 模型定义
│   │   ├── lite_digit_detector.py  # 数字检测模型
│   ├── capture_processing.py  # 图像捕获和处理
│   ├── capture_guideline.py   # 引导线捕获
│   ├── input_monitor.py    # 输入监控
│   └── controller_output.py  # 控制器输出
├── model_tools/            # 模型训练工具
│   ├── build_digit_datasets.py  # 构建数字数据集
│   └── train_digit_detector.py  # 训练数字检测器
├── model/                  # 预训练模型存储
│   └── LDD/                # 轻量级数字检测器模型
├── dataset/                # 数据集
│   └── digit/              # 数字检测数据集
├── raw_data/               # 原始数据
│   └── digit/              # 原始数字图像
├── run/                    # 运行时生成的文件
├── example/                # 示例脚本
└── README.md               # 项目说明
```

## 数据采集

在训练自动驾驶模型之前，我们需要收集游戏中的图像数据和对应的控制输入。

### 屏幕捕获数据

1. 首先，确保游戏以窗口或全屏模式运行。

2. 使用我们的屏幕捕获工具来捕获游戏画面：

```python
from utils.vendor.grabscreen import ScreenGrabber
from utils.capture_processing import ImgHelper
import time
import cv2
import os

# 创建保存数据的目录
if not os.path.exists("raw_data"):
    os.makedirs("raw_data")
if not os.path.exists("raw_data/digit"):
    os.makedirs("raw_data/digit")

# 初始化屏幕捕获
grabber = ScreenGrabber()

# 捕获一系列图像
for i in range(1000):  # 捕获1000张图像
    frame = grabber.grab()
    # 保存原始屏幕截图
    ImgHelper.save_screenshot(frame, f"raw_data/digit/{i}.jpg")
    print(f"已保存第 {i+1}/1000 张图像")
    time.sleep(0.5)  # 每0.5秒捕获一次
```

### 控制器输入监控

如果你想收集人类驾驶数据进行模仿学习，可以使用我们的输入监控工具：

```python
from utils.input_monitor import InputMonitor
from utils.vendor.grabscreen import ScreenGrabber
import time
import cv2
import os
import json

# 创建保存数据的目录
if not os.path.exists("raw_data/driving"):
    os.makedirs("raw_data/driving")

# 初始化屏幕捕获和输入监控
grabber = ScreenGrabber()
monitor = InputMonitor()
monitor.start_controller_monitoring()  # 或 start_keyboard_monitoring()

try:
    # 同时捕获屏幕和控制器输入
    for i in range(1000):
        # 捕获屏幕
        frame = grabber.grab()
        
        # 获取控制器状态
        control_state = monitor.get_combined_state()
        
        # 保存图像
        cv2.imwrite(f"raw_data/driving/img_{i:05d}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # 保存控制数据
        with open(f"raw_data/driving/control_{i:05d}.json", "w") as f:
            json.dump(control_state, f)
            
        print(f"已保存第 {i+1}/1000 帧数据")
        time.sleep(0.1)  # 每0.1秒捕获一次
        
except KeyboardInterrupt:
    print("数据收集已停止")
finally:
    monitor.stop_monitoring()
```

## 模型训练

### 数字识别模型训练

这部分训练一个轻量级模型来识别游戏中显示的速度数字：

1. 首先，我们需要处理原始图像数据并构建训练数据集：

```bash
python model_tools/build_digit_datasets.py
```

这个脚本会处理 `raw_data/digit` 中的图像，裁剪出速度数字区域，使用 EasyOCR 初步识别并生成标签，最后将处理好的数据集保存到 `dataset/digit` 目录。

2. 接下来，训练数字识别模型：

```bash
python model_tools/train_digit_detector.py
```

这个脚本将训练 `LiteDigitDetector` 模型，该模型能够识别游戏中的三位速度数字。训练完成后，最佳模型将保存到 `run/light_digit_detector/best_digit_model.pth`。

3. 测试模型效果：

```bash
python example/digit_detection_example.py
```

### 模型示例

您可以查看 `example` 目录中的各种示例来了解如何使用我们的工具：

- `controller_input_monitor_example.py`: 演示如何监控游戏控制器输入
- `keyboard_input_monitor_example.py`: 演示如何监控键盘输入
- `controller_output_example.py`: 演示如何通过虚拟控制器控制游戏
- `digit_detection_example.py`: 演示如何使用训练好的数字检测模型

## 运行自动驾驶

自动驾驶功能还在开发中，敬请期待...

## 致谢

本项目基于以下开源项目的一些代码和思路：
- [@蔡俊志](https://github.com/EthanNCai/AI-Plays-ForzaHorizon4) - 屏幕捕获代码
- [@Dedsecer](https://github.com/DedSecer/AI-Plays-ForzaHorizon4) - 图像处理功能

感谢上述项目作者的贡献！
