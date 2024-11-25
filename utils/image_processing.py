 """
CT图像预处理模块: 提供CT图像的加载、预处理和增强功能
"""
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Union, Tuple, List, Optional

from config import DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH


def load_image(image_path: str) -> np.ndarray:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        加载的图像数组
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 使用PIL加载图像并转换为numpy数组
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # 如果是RGB图像，转换为灰度图
    if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    return image_array


def resize_image(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    调整图像尺寸
    
    Args:
        image: 输入图像
        size: 目标尺寸 (高度, 宽度)
        
    Returns:
        调整尺寸后的图像
    """
    return cv2.resize(image, size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    标准化图像，使其值范围在0-1之间
    
    Args:
        image: 输入图像
        
    Returns:
        标准化后的图像
    """
    if image.max() == image.min():
        return np.zeros_like(image, dtype=np.float32)
    
    normalized = (image - image.min()) / (image.max() - image.min())
    return normalized


def apply_window_level(
    image: np.ndarray, 
    window_center: int = DEFAULT_WINDOW_CENTER, 
    window_width: int = DEFAULT_WINDOW_WIDTH
) -> np.ndarray:
    """
    应用窗宽窗位调整，用于CT图像的对比度优化
    
    Args:
        image: 输入CT图像
        window_center: 窗位(WL)
        window_width: 窗宽(WW)
        
    Returns:
        调整后的图像
    """
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    
    windowed = np.clip(image, min_value, max_value)
    windowed = (windowed - min_value) / (max_value - min_value)
    
    return windowed


def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """
    增强图像对比度
    
    Args:
        image: 输入图像
        alpha: 对比度控制参数
        beta: 亮度控制参数
        
    Returns:
        对比度增强后的图像
    """
    # 确保图像值在0-1范围内
    if image.max() > 1.0:
        image = normalize_image(image)
    
    # 应用对比度增强
    enhanced = np.clip(alpha * image + beta, 0, 1)
    return enhanced


def denoise_image(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """
    降噪处理
    
    Args:
        image: 输入图像
        strength: 降噪强度
        
    Returns:
        降噪后的图像
    """
    # 确保图像是正确的类型和范围
    if image.max() <= 1.0:
        image_8bit = (image * 255).astype(np.uint8)
    else:
        image_8bit = image.astype(np.uint8)
    
    # 应用非局部均值去噪
    denoised = cv2.fastNlMeansDenoising(image_8bit, None, strength, 7, 21)
    
    # 如果原图是浮点型，返回浮点型结果
    if image.max() <= 1.0:
        return denoised / 255.0
    
    return denoised


def preprocess_for_biomedclip(image_path: str) -> np.ndarray:
    """
    预处理图像以适应BiomedCLIP模型输入要求
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        预处理后的图像数组
    """
    # 加载图像
    image = load_image(image_path)
    
    # 应用窗宽窗位调整
    image = apply_window_level(image)
    
    # 调整大小为模型输入尺寸
    image = resize_image(image, (224, 224))
    
    # 转换为PIL图像以应用torchvision转换
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    
    # 应用BiomedCLIP预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform(image_pil).numpy()


def process_ct_batch(image_paths: List[str]) -> List[np.ndarray]:
    """
    批量处理多个CT图像
    
    Args:
        image_paths: CT图像路径列表
        
    Returns:
        处理后的图像列表
    """
    processed_images = []
    for image_path in image_paths:
        try:
            processed_image = preprocess_for_biomedclip(image_path)
            processed_images.append(processed_image)
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
    
    return processed_images
