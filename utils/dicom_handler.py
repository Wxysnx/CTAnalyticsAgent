 """
DICOM处理模块: 提供DICOM医学影像文件的读取和处理功能
"""
import os
import pydicom
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from config import DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH


def load_dicom(dicom_path: str) -> pydicom.FileDataset:
    """
    加载DICOM文件
    
    Args:
        dicom_path: DICOM文件路径
        
    Returns:
        DICOM数据集对象
    """
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOM文件不存在: {dicom_path}")
    
    return pydicom.dcmread(dicom_path)


def extract_dicom_metadata(dicom_data: pydicom.FileDataset) -> Dict[str, Any]:
    """
    提取DICOM元数据信息
    
    Args:
        dicom_data: DICOM数据集
        
    Returns:
        包含关键元数据的字典
    """
    metadata = {}
    
    # 尝试提取常见的DICOM标签
    try:
        if hasattr(dicom_data, 'PatientID'):
            metadata['PatientID'] = dicom_data.PatientID
        if hasattr(dicom_data, 'PatientName'):
            metadata['PatientName'] = str(dicom_data.PatientName)
        if hasattr(dicom_data, 'PatientBirthDate'):
            metadata['PatientBirthDate'] = dicom_data.PatientBirthDate
        if hasattr(dicom_data, 'PatientSex'):
            metadata['PatientSex'] = dicom_data.PatientSex
        if hasattr(dicom_data, 'StudyDate'):
            metadata['StudyDate'] = dicom_data.StudyDate
        if hasattr(dicom_data, 'StudyDescription'):
            metadata['StudyDescription'] = dicom_data.StudyDescription
        if hasattr(dicom_data, 'Modality'):
            metadata['Modality'] = dicom_data.Modality
        if hasattr(dicom_data, 'BodyPartExamined'):
            metadata['BodyPartExamined'] = dicom_data.BodyPartExamined
        if hasattr(dicom_data, 'SliceThickness'):
            metadata['SliceThickness'] = dicom_data.SliceThickness
        if hasattr(dicom_data, 'WindowCenter'):
            metadata['WindowCenter'] = dicom_data.WindowCenter
        if hasattr(dicom_data, 'WindowWidth'):
            metadata['WindowWidth'] = dicom_data.WindowWidth
    except Exception as e:
        print(f"提取元数据时出错: {e}")
    
    return metadata


def dicom_to_numpy(dicom_data: pydicom.FileDataset) -> np.ndarray:
    """
    将DICOM数据转换为NumPy数组
    
    Args:
        dicom_data: DICOM数据集
        
    Returns:
        表示图像的NumPy数组
    """
    # 提取像素数据
    image = dicom_data.pixel_array.astype(np.float32)
    
    # 应用放射度变换
    if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
        image = image * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
    
    return image


def window_dicom_image(
    image: np.ndarray, 
    dicom_data: Optional[pydicom.FileDataset] = None, 
    window_center: Optional[int] = None,
    window_width: Optional[int] = None
) -> np.ndarray:
    """
    应用窗宽窗位设置调整DICOM图像
    
    Args:
        image: 图像数组
        dicom_data: DICOM数据集，用于获取默认窗宽窗位
        window_center: 自定义窗位，优先级高于DICOM标签
        window_width: 自定义窗宽，优先级高于DICOM标签
        
    Returns:
        调整后的图像
    """
    # 确定窗宽窗位值
    if window_center is None and window_width is None and dicom_data is not None:
        if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
            window_center = dicom_data.WindowCenter
            window_width = dicom_data.WindowWidth
            
            # 处理多值情况
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]
        else:
            window_center = DEFAULT_WINDOW_CENTER
            window_width = DEFAULT_WINDOW_WIDTH
    elif window_center is None or window_width is None:
        window_center = window_center or DEFAULT_WINDOW_CENTER
        window_width = window_width or DEFAULT_WINDOW_WIDTH
    
    # 计算窗口的最小值和最大值
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    
    # 应用窗宽窗位
    windowed = np.clip(image, min_value, max_value)
    
    # 标准化到0-1范围
    if max_value != min_value:  # 防止除以零
        windowed = (windowed - min_value) / (max_value - min_value)
    else:
        windowed = np.zeros_like(windowed)
    
    return windowed


def save_dicom_as_png(dicom_path: str, output_path: str, apply_window: bool = True) -> str:
    """
    将DICOM文件转换为PNG图像并保存
    
    Args:
        dicom_path: DICOM文件路径
        output_path: 输出PNG文件路径
        apply_window: 是否应用默认窗宽窗位
        
    Returns:
        保存的PNG文件路径
    """
    import cv2
    
    # 加载DICOM
    dicom_data = load_dicom(dicom_path)
    
    # 转换为NumPy数组
    image = dicom_to_numpy(dicom_data)
    
    # 应用窗宽窗位
    if apply_window:
        image = window_dicom_image(image, dicom_data)
    else:
        # 如果不应用窗宽窗位，则进行线性标准化
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # 转换为8位图像
    image_8bit = (image * 255).astype(np.uint8)
    
    # 保存为PNG
    cv2.imwrite(output_path, image_8bit)
    
    return output_path


def process_dicom_directory(
    dicom_dir: str, 
    output_dir: str, 
    extension: str = ".dcm"
) -> List[str]:
    """
    处理目录中的所有DICOM文件
    
    Args:
        dicom_dir: DICOM文件目录
        output_dir: 输出PNG文件目录
        extension: DICOM文件扩展名
        
    Returns:
        处理后的PNG文件路径列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个DICOM文件
    png_paths = []
    for filename in os.listdir(dicom_dir):
        if filename.lower().endswith(extension):
            dicom_path = os.path.join(dicom_dir, filename)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(output_dir, png_filename)
            
            try:
                save_dicom_as_png(dicom_path, png_path)
                png_paths.append(png_path)
            except Exception as e:
                print(f"处理文件 {dicom_path} 时出错: {e}")
    
    return png_paths
