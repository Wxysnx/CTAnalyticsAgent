 """
报告生成工具: 生成专业医疗诊断报告
"""
from typing import Dict, Any, List, Optional, Union
import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import REPORT_TEMPLATE
from utils.report_formatter import (
    format_ct_report,
    save_report_to_markdown,
    save_report_to_json
)


class MedicalReportGenerator:
    """医疗报告生成器"""
    
    def __init__(self, llm: BaseLanguageModel):
        """
        初始化医疗报告生成器
        
        Args:
            llm: 语言模型，用于生成报告内容
        """
        self.llm = llm
    
    def generate_ct_report(
        self,
        ct_analysis: Dict[str, Any],
        medical_knowledge: str,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        生成CT诊断报告
        
        Args:
            ct_analysis: CT分析结果
            medical_knowledge: 医学知识上下文
            output_format: 输出格式 ("markdown", "json", "dict")
            
        Returns:
            包含报告内容的字典
        """
        # 提取CT分析描述
        if "summary" in ct_analysis:
            ct_description = ct_analysis["summary"].get("combined_description", "")
            scan_region = ct_analysis["summary"].get("scan_region", "未知区域")
        elif "analysis" in ct_analysis:
            ct_description = ct_analysis["analysis"].get("combined_description", "")
            scan_region = "未知区域"
        else:
            ct_description = str(ct_analysis)
            scan_region = "未知区域"
        
        # 创建提示模板
        template = """
        作为一名经验丰富的放射科医师，请基于下面的CT图像分析结果和相关医学知识，生成一份专业的CT诊断报告。
        你的报告应当包含影像发现、分析与解释、诊断意见和建议等部分。
        
        ## CT图像分析结果
        {ct_description}
        
        ## 相关医学知识
        {medical_knowledge}
        
        请提供以下四个部分的内容，每部分都要详细专业：
        
        1. 影像发现: (详细描述CT图像中观察到的客观发现)
        
        2. 分析与解释: (对影像发现进行专业分析和解释)
        
        3. 诊断意见: (给出可能的诊断，如有多种可能，请列出并标明优先级)
        
        4. 建议: (提供针对诊断的下一步建议，如进一步检查、随访或治疗方案)
        
        请确保报告内容专业准确，使用医学术语，但同时要清晰易懂。
        """
        
        # 创建提示
        prompt = PromptTemplate(
            template=template,
            input_variables=["ct_description", "medical_knowledge"]
        )
        
        # 创建链
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # 执行链
        result = chain.run(
            ct_description=ct_description,
            medical_knowledge=medical_knowledge
        )
        
        # 提取各个部分
        report_sections = self._extract_report_sections(result)
        
        # 生成报告
        if output_format == "markdown" or output_format == "md":
            report_content = format_ct_report(
                image_findings=report_sections["影像发现"],
                analysis=report_sections["分析与解释"],
                diagnostic_opinion=report_sections["诊断意见"],
                recommendations=report_sections["建议"],
                examination_area=scan_region
            )
            report_data = {"content": report_content, "format": "markdown"}
        else:
            report_data = {
                "image_findings": report_sections["影像发现"],
                "analysis": report_sections["分析与解释"],
                "diagnostic_opinion": report_sections["诊断意见"],
                "recommendations": report_sections["建议"],
                "examination_area": scan_region,
                "examination_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "report_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "format": output_format
            }
        
        return report_data
    
    def _extract_report_sections(self, report_text: str) -> Dict[str, str]:
        """
        从生成的文本中提取报告各个部分
        
        Args:
            report_text: 生成的报告文本
            
        Returns:
            报告各部分内容的字典
        """
        sections = {
            "影像发现": "",
            "分析与解释": "",
            "诊断意见": "",
            "建议": ""
        }
        
        # 查找各部分的位置
        section_markers = [
            ("影像发现", ["1. 影像发现", "影像发现:", "影像发现："]),
            ("分析与解释", ["2. 分析与解释", "分析与解释:", "分析与解释："]),
            ("诊断意见", ["3. 诊断意见", "诊断意见:", "诊断意见："]),
            ("建议", ["4. 建议", "建议:", "建议："])
        ]
        
        # 初始化每个部分的起始位置
        positions = {}
        
        # 寻找每个部分的起始位置
        for section, markers in section_markers:
            for marker in markers:
                pos = report_text.find(marker)
                if pos >= 0:
                    # 加上标记的长度，跳过标记本身
                    positions[section] = pos + len(marker)
                    break
        
        # 按位置排序节段
        sorted_sections = sorted(positions.items(), key=lambda x: x[1])
        
        # 提取每个部分的内容
        for i, (section, start) in enumerate(sorted_sections):
            # 如果不是最后一个部分，则截取到下一个部分开始位置
            if i < len(sorted_sections) - 1:
                next_section, next_start = sorted_sections[i+1]
                section_content = report_text[start:next_start].strip()
            else:
                # 最后一个部分，截取到文本末尾
                section_content = report_text[start:].strip()
            
            sections[section] = section_content
        
        return sections
    
    def save_report(
        self,
        report_data: Dict[str, Any],
        output_path: str,
        format_type: Optional[str] = None
    ) -> str:
        """
        保存报告到文件
        
        Args:
            report_data: 报告数据
            output_path: 输出路径
            format_type: 输出格式类型，如果不指定则从report_data中获取
            
        Returns:
            保存的文件路径
        """
        # 确定输出格式
        if format_type is None:
            format_type = report_data.get("format", "markdown")
        
        # 根据格式保存
        if format_type == "markdown" or format_type == "md":
            if "content" in report_data:
                content = report_data["content"]
            else:
                content = format_ct_report(
                    image_findings=report_data.get("image_findings", ""),
                    analysis=report_data.get("analysis", ""),
                    diagnostic_opinion=report_data.get("diagnostic_opinion", ""),
                    recommendations=report_data.get("recommendations", ""),
                    examination_area=report_data.get("examination_area", "未知区域"),
                    examination_date=report_data.get("examination_date"),
                    report_date=report_data.get("report_date")
                )
            return save_report_to_markdown(content, output_path)
        else:
            return save_report_to_json(report_data, output_path)
    
    def generate_comparative_report(
        self,
        current_ct_analysis: Dict[str, Any],
        previous_ct_analysis: Dict[str, Any],
        medical_knowledge: str
    ) -> Dict[str, Any]:
        """
        生成比较性CT诊断报告，对比当前与之前的扫描结果
        
        Args:
            current_ct_analysis: 当前CT分析结果
            previous_ct_analysis: 之前的CT分析结果
            medical_knowledge: 医学知识上下文
            
        Returns:
            包含比较报告内容的字典
        """
        # 提取CT分析描述
        if "summary" in current_ct_analysis:
            current_description = current_ct_analysis["summary"].get("combined_description", "")
        elif "analysis" in current_ct_analysis:
            current_description = current_ct_analysis["analysis"].get("combined_description", "")
        else:
            current_description = str(current_ct_analysis)
        
        if "summary" in previous_ct_analysis:
            previous_description = previous_ct_analysis["summary"].get("combined_description", "")
        elif "analysis" in previous_ct_analysis:
            previous_description = previous_ct_analysis["analysis"].get("combined_description", "")
        else:
            previous_description = str(previous_ct_analysis)
        
        # 创建提示模板
        template = """
        作为一名经验丰富的放射科医师，请比较当前和之前的CT扫描结果，并生成一份对比分析报告。
        
        ## 当前CT结果
        {current_description}
        
        ## 之前CT结果
        {previous_description}
        
        ## 相关医学知识
        {medical_knowledge}
        
        请提供以下四个部分的内容：
        
        1. 影像对比发现: (对比当前和之前CT的差异，包括新发现和变化)
        
        2. 分析与解释: (对变化的专业分析和解释)
        
        3. 诊断意见: (基于对比结果给出诊断意见，说明疾病进展情况)
        
        4. 建议: (提供针对对比结果的建议)
        
        请使用专业医学术语，同时保持清晰易懂。
        """
        
        # 创建提示
        prompt = PromptTemplate(
            template=template,
            input_variables=["current_description", "previous_description", "medical_knowledge"]
        )
        
        # 创建链
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # 执行链
        result = chain.run(
            current_description=current_description,
            previous_description=previous_description,
            medical_knowledge=medical_knowledge
        )
        
        # 提取各个部分
        report_sections = self._extract_report_sections(result)
        
        # 设置扫描区域
        scan_region = "未知区域"
        if "summary" in current_ct_analysis:
            scan_region = current_ct_analysis["summary"].get("scan_region", scan_region)
        
        # 生成报告
        report_content = format_ct_report(
            image_findings=report_sections["影像发现"],
            analysis=report_sections["分析与解释"],
            diagnostic_opinion=report_sections["诊断意见"],
            recommendations=report_sections["建议"],
            examination_area=scan_region,
            additional_info={"报告类型": "对比分析报告"}
        )
        
        return {"content": report_content, "format": "markdown"}
