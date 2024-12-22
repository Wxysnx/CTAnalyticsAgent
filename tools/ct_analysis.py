 """
CT图像分析工具: 使用BiomedCLIP模型分析CT图像
"""
import os
import json
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from config import BIOMEDCLIP_MODEL_NAME
from utils.image_processing import preprocess_for_biomedclip, load_image, apply_window_level
from utils.dicom_handler import load_dicom, dicom_to_numpy, extract_dicom_metadata


class BiomedCLIPTool:
    """使用BiomedCLIP模型分析CT图像的工具"""
    
    def __init__(
        self, 
        model_name: str = BIOMEDCLIP_MODEL_NAME,
        device: Optional[str] = None
    ):
        """
        初始化BiomedCLIPTool
        
        Args:
            model_name: BiomedCLIP模型名称
            device: 计算设备 ('cpu', 'cuda', 'mps')
        """
        # 确定设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载BiomedCLIP处理器和模型
        try:
            print(f"加载BiomedCLIP模型: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            print("BiomedCLIP模型加载完成")
        except Exception as e:
            print(f"加载BiomedCLIP模型时出错: {e}")
            raise
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        分析CT图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含分析结果的字典
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 确定文件类型
        is_dicom = image_path.lower().endswith('.dcm')
        
        # 处理DICOM文件
        if is_dicom:
            return self._analyze_dicom_image(image_path)
        # 处理常规图像文件
        else:
            return self._analyze_regular_image(image_path)
    
    def _analyze_dicom_image(self, dicom_path: str) -> Dict[str, Any]:
        """
        分析DICOM格式CT图像
        
        Args:
            dicom_path: DICOM文件路径
            
        Returns:
            包含分析结果的字典
        """
        try:
            # 加载DICOM文件
            dicom_data = load_dicom(dicom_path)
            
            # 提取元数据
            metadata = extract_dicom_metadata(dicom_data)
            
            # 转换为NumPy数组
            image_array = dicom_to_numpy(dicom_data)
            
            # 应用窗宽窗位
            if 'WindowCenter' in metadata and 'WindowWidth' in metadata:
                window_center = float(metadata['WindowCenter'])
                window_width = float(metadata['WindowWidth'])
                processed_image = apply_window_level(image_array, window_center, window_width)
            else:
                processed_image = apply_window_level(image_array)
            
            # 使用BiomedCLIP分析图像
            analysis_result = self._run_biomedclip_analysis(processed_image)
            
            # 合并结果
            result = {
                "metadata": metadata,
                "analysis": analysis_result,
                "image_type": "DICOM"
            }
            
            return result
            
        except Exception as e:
            print(f"分析DICOM图像时出错: {e}")
            raise
    
    def _analyze_regular_image(self, image_path: str) -> Dict[str, Any]:
        """
        分析常规格式CT图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含分析结果的字典
        """
        try:
            # 预处理图像
            processed_image = preprocess_for_biomedclip(image_path)
            
            # 使用BiomedCLIP分析图像
            analysis_result = self._run_biomedclip_analysis(processed_image)
            
            # 构建结果
            result = {
                "metadata": {
                    "filename": os.path.basename(image_path),
                    "file_type": os.path.splitext(image_path)[1][1:]
                },
                "analysis": analysis_result,
                "image_type": "Regular"
            }
            
            return result
            
        except Exception as e:
            print(f"分析常规图像时出错: {e}")
            raise
    
    def _run_biomedclip_analysis(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        运行BiomedCLIP模型分析
        
        Args:
            image_data: 预处理后的图像数据
            
        Returns:
            分析结果字典
        """
        # 准备医学描述模板
        medical_descriptions = [
            "正常胸部CT图像，无明显异常。",
            "肺部有磨玻璃样阴影，考虑肺炎可能。",
            "肺部有结节影，需要进一步评估。",
            "肺部有实质性浸润影，考虑感染或肿瘤。",
            "胸腔积液，肺实质无明显异常。",
            "肺气肿表现，肺部透明度增高。",
            "肺间质改变，考虑间质性肺病。",
            "支气管扩张，有蜂窝状改变。",
            "肺内占位性病变，需要进一步评估。",
            "纵隔淋巴结肿大，考虑炎症或肿瘤。",
            "肺动脉高压表现，肺动脉主干增宽。",
            "冠状动脉钙化，考虑冠心病。"
        ]
        
        try:
            # 转换图像格式以适应模型输入
            if isinstance(image_data, np.ndarray):
                if image_data.ndim == 2:  # 单通道图像
                    image_pil = Image.fromarray((image_data * 255).astype(np.uint8))
                else:  # 多通道图像
                    image_pil = Image.fromarray((image_data[0] * 255).astype(np.uint8))
            else:
                raise ValueError("图像数据格式不支持")
            
            # 使用处理器准备输入
            inputs = self.processor(
                text=medical_descriptions,
                images=image_pil,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # 执行推理
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 计算图像文本相似度
            image_embeds = outputs.vision_model_output.pooler_output
            text_embeds = outputs.text_model_output.pooler_output
            
            # 归一化嵌入
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # 计算相似度分数
            logits_per_image = image_embeds @ text_embeds.t()
            probs = torch.nn.functional.softmax(logits_per_image, dim=-1)
            
            # 将结果转换为CPU并提取概率
            probs_list = probs.cpu().numpy().tolist()[0]
            
            # 整理结果
            results = []
            for desc, prob in zip(medical_descriptions, probs_list):
                results.append({
                    "description": desc,
                    "probability": prob
                })
            
            # 按概率排序
            results = sorted(results, key=lambda x: x["probability"], reverse=True)
            
            # 生成综合描述
            top_descriptions = [r["description"] for r in results[:3]]
            combined_description = self._generate_combined_description(top_descriptions)
            
            # 构建最终结果
            analysis_result = {
                "top_matches": results[:5],
                "combined_description": combined_description,
                "abnormality_detected": "正常" not in results[0]["description"],
                "confidence": results[0]["probability"]
            }
            
            return analysis_result
            
        except Exception as e:
            print(f"运行BiomedCLIP分析时出错: {e}")
            raise
    
    def _generate_combined_description(self, top_descriptions: List[str]) -> str:
        """
        生成综合描述
        
        Args:
            top_descriptions: 概率最高的几个描述
            
        Returns:
            综合描述文本
        """
        # 如果第一个描述是正常的，而且概率很高，直接返回
        if "正常" in top_descriptions[0]:
            return top_descriptions[0]
        
        # 否则，组合前三个描述
        combined = "CT图像分析显示："
        
        for i, desc in enumerate(top_descriptions):
            if i == 0:
                combined += desc
            else:
                # 移除句首，只保留关键信息
                cleaned_desc = desc.split("，", 1)[-1] if "，" in desc else desc
                combined += f"；另外可能{cleaned_desc}"
        
        return combined

    def analyze_multiple_images(
        self, 
        image_paths: List[str]
    ) -> Dict[str, Any]:
        """
        分析多张CT图像并汇总结果
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            汇总的分析结果
        """
        # 存储每张图像的分析结果
        individual_results = []
        
        # 分析每张图像
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path)
                individual_results.append(result)
            except Exception as e:
                print(f"分析图像 {image_path} 时出错: {e}")
        
        # 汇总结果
        summary = self._summarize_analysis_results(individual_results)
        
        # 构建完整结果
        return {
            "summary": summary,
            "individual_results": individual_results,
            "image_count": len(individual_results)
        }
    
    def _summarize_analysis_results(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        汇总多个分析结果
        
        Args:
            results: 分析结果列表
            
        Returns:
            汇总结果
        """
        if not results:
            return {"error": "无有效分析结果"}
        
        # 收集所有描述
        all_descriptions = []
        abnormality_detected = False
        
        for result in results:
            analysis = result.get("analysis", {})
            if analysis:
                all_descriptions.append(analysis.get("combined_description", ""))
                if analysis.get("abnormality_detected", False):
                    abnormality_detected = True
        
        # 简单合并所有描述
        combined_description = " ".join(all_descriptions)
        
        # 构建汇总结果
        summary = {
            "combined_description": combined_description,
            "abnormality_detected": abnormality_detected,
            "scan_region": self._determine_scan_region(results)
        }
        
        return summary
    
    def _determine_scan_region(self, results: List[Dict[str, Any]]) -> str:
        """
        确定扫描区域
        
        Args:
            results: 分析结果列表
            
        Returns:
            扫描区域描述
        """
        # 从DICOM元数据中提取扫描区域信息
        for result in results:
            metadata = result.get("metadata", {})
            if "BodyPartExamined" in metadata:
                return metadata["BodyPartExamined"]
        
        # 如果没有找到，尝试从描述中猜测
        descriptions = []
        for result in results:
            if "analysis" in result and "combined_description" in result["analysis"]:
                descriptions.append(result["analysis"]["combined_description"])
        
        combined_text = " ".join(descriptions).lower()
        
        # 简单规则匹配
        if "胸" in combined_text or "肺" in combined_text:
            return "胸部"
        elif "腹" in combined_text or "肝" in combined_text or "脾" in combined_text:
            return "腹部"
        elif "头" in combined_text or "脑" in combined_text:
            return "头部"
        else:
            return "未知区域"


def create_ct_analysis_tool() -> BiomedCLIPTool:
    """
    创建CT分析工具实例
    
    Returns:
        BiomedCLIPTool实例
    """
    return BiomedCLIPTool()


def analyze_ct_images(image_paths: Union[str, List[str]]) -> Dict[str, Any]:
    """
    分析CT图像的便捷函数
    
    Args:
        image_paths: 单个图像路径或图像路径列表
        
    Returns:
        分析结果
    """
    tool = create_ct_analysis_tool()
    
    if isinstance(image_paths, str):
        return tool.analyze_image(image_paths)
    else:
        return tool.analyze_multiple_images(image_paths)


def save_analysis_result(result: Dict[str, Any], output_path: str) -> None:
    """
    保存分析结果到JSON文件
    
    Args:
        result: 分析结果
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
