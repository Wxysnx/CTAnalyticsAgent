 """
报告格式化模块: 提供医学报告格式化和处理功能
"""
import json
import datetime
from typing import Dict, Any, List, Optional

from config import REPORT_TEMPLATE


def format_ct_report(
    image_findings: str,
    analysis: str,
    diagnostic_opinion: str,
    recommendations: str,
    examination_area: str = "胸部/腹部",
    examination_date: Optional[str] = None,
    report_date: Optional[str] = None,
    additional_info: Dict[str, Any] = None
) -> str:
    """
    格式化CT诊断报告
    
    Args:
        image_findings: 影像发现内容
        analysis: 分析与解释内容
        diagnostic_opinion: 诊断意见内容
        recommendations: 建议内容
        examination_area: 检查部位
        examination_date: 检查日期(可选)，默认为当前日期
        report_date: 报告日期(可选)，默认为当前日期
        additional_info: 其他额外信息(可选)
        
    Returns:
        格式化后的报告文本
    """
    # 处理日期
    if examination_date is None:
        examination_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if report_date is None:
        report_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 准备报告内容
    report_content = REPORT_TEMPLATE.format(
        examination_date=examination_date,
        examination_area=examination_area,
        image_findings=image_findings,
        analysis_and_interpretation=analysis,
        diagnostic_opinion=diagnostic_opinion,
        recommendations=recommendations,
        report_date=report_date
    )
    
    # 如果有额外信息，添加到报告末尾
    if additional_info:
        additional_section = "\n## 附加信息\n"
        for key, value in additional_info.items():
            additional_section += f"- **{key}**: {value}\n"
        report_content += additional_section
    
    return report_content


def save_report_to_markdown(report_content: str, output_path: str) -> str:
    """
    将报告保存为Markdown文件
    
    Args:
        report_content: 报告内容
        output_path: 输出文件路径
        
    Returns:
        保存的文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(report_content)
    
    return output_path


def save_report_to_json(
    report_data: Dict[str, Any], 
    output_path: str, 
    indent: int = 2
) -> str:
    """
    将报告数据保存为JSON文件
    
    Args:
        report_data: 报告数据字典
        output_path: 输出文件路径
        indent: JSON缩进空格数
        
    Returns:
        保存的文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(report_data, file, ensure_ascii=False, indent=indent)
    
    return output_path


def report_data_to_markdown(report_data: Dict[str, Any]) -> str:
    """
    将报告数据字典转换为Markdown格式
    
    Args:
        report_data: 报告数据字典
        
    Returns:
        Markdown格式的报告内容
    """
    # 提取关键信息
    image_findings = report_data.get('image_findings', '')
    analysis = report_data.get('analysis', '')
    diagnostic_opinion = report_data.get('diagnostic_opinion', '')
    recommendations = report_data.get('recommendations', '')
    examination_area = report_data.get('examination_area', '胸部/腹部')
    examination_date = report_data.get('examination_date')
    report_date = report_data.get('report_date')
    
    # 提取额外信息
    additional_info = {}
    for key, value in report_data.items():
        if key not in ['image_findings', 'analysis', 'diagnostic_opinion', 
                       'recommendations', 'examination_area', 
                       'examination_date', 'report_date']:
            additional_info[key] = value
    
    # 生成并返回格式化报告
    return format_ct_report(
        image_findings=image_findings,
        analysis=analysis,
        diagnostic_opinion=diagnostic_opinion,
        recommendations=recommendations,
        examination_area=examination_area,
        examination_date=examination_date,
        report_date=report_date,
        additional_info=additional_info
    )


def markdown_to_report_data(markdown_content: str) -> Dict[str, Any]:
    """
    从Markdown格式报告中提取数据
    
    Args:
        markdown_content: Markdown格式的报告内容
        
    Returns:
        报告数据字典
    """
    report_data = {}
    
    # 提取检查日期
    import re
    examination_date_match = re.search(r'\*\*检查日期\*\*:\s*(.*)', markdown_content)
    if examination_date_match:
        report_data['examination_date'] = examination_date_match.group(1).strip()
    
    # 提取检查部位
    examination_area_match = re.search(r'\*\*检查部位\*\*:\s*(.*)', markdown_content)
    if examination_area_match:
        report_data['examination_area'] = examination_area_match.group(1).strip()
    
    # 提取影像发现
    findings_start = markdown_content.find('## 影像发现')
    analysis_start = markdown_content.find('## 分析与解释')
    if findings_start >= 0 and analysis_start >= 0:
        findings_text = markdown_content[findings_start+len('## 影像发现'):analysis_start].strip()
        report_data['image_findings'] = findings_text
    
    # 提取分析与解释
    analysis_start = markdown_content.find('## 分析与解释')
    diagnosis_start = markdown_content.find('## 诊断意见')
    if analysis_start >= 0 and diagnosis_start >= 0:
        analysis_text = markdown_content[analysis_start+len('## 分析与解释'):diagnosis_start].strip()
        report_data['analysis'] = analysis_text
    
    # 提取诊断意见
    diagnosis_start = markdown_content.find('## 诊断意见')
    recommendations_start = markdown_content.find('## 建议')
    if diagnosis_start >= 0 and recommendations_start >= 0:
        diagnosis_text = markdown_content[diagnosis_start+len('## 诊断意见'):recommendations_start].strip()
        report_data['diagnostic_opinion'] = diagnosis_text
    
    # 提取建议
    recommendations_start = markdown_content.find('## 建议')
    report_date_start = markdown_content.find('## 报告日期')
    if recommendations_start >= 0 and report_date_start >= 0:
        recommendations_text = markdown_content[recommendations_start+len('## 建议'):report_date_start].strip()
        report_data['recommendations'] = recommendations_text
    
    # 提取报告日期
    report_date_start = markdown_content.find('## 报告日期')
    additional_info_start = markdown_content.find('## 附加信息')
    if report_date_start >= 0:
        if additional_info_start >= 0:
            report_date_text = markdown_content[report_date_start+len('## 报告日期'):additional_info_start].strip()
        else:
            report_date_text = markdown_content[report_date_start+len('## 报告日期'):].strip()
        report_data['report_date'] = report_date_text
    
    return report_data


def append_to_report(
    original_report: str, 
    section_title: str, 
    content: str
) -> str:
    """
    向报告中追加新内容
    
    Args:
        original_report: 原始报告内容
        section_title: 要添加的章节标题
        content: 要添加的内容
        
    Returns:
        更新后的报告内容
    """
    # 检查是否已存在该章节
    section_header = f"## {section_title}"
    
    if section_header in original_report:
        # 如果章节已存在，更新内容
        import re
        pattern = f"(## {section_title}.*?)(?=## |$)"
        replacement = f"## {section_title}\n{content}\n\n"
        updated_report = re.sub(pattern, replacement, original_report, flags=re.DOTALL)
        return updated_report
    else:
        # 如果章节不存在，添加到末尾
        return f"{original_report.rstrip()}\n\n## {section_title}\n{content}\n"
