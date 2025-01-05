 """
定义CrewAI任务: CT图像分析、知识检索和报告生成任务
"""
from typing import Dict, Any, List, Optional, Union
import os

from crewai import Task
from crewai.agent import Agent

from tools.ct_analysis import BiomedCLIPTool
from tools.knowledge_retrieval import MedicalKnowledgeRetrievalTool
from tools.report_generation import MedicalReportGenerator


def create_image_analysis_task(
    agent: Agent,
    ct_analysis_tool: BiomedCLIPTool,
    image_paths: Union[str, List[str]],
    task_id: str = "image_analysis"
) -> Task:
    """
    创建CT图像分析任务
    
    Args:
        agent: 分配任务的智能体
        ct_analysis_tool: CT分析工具
        image_paths: CT图像路径或路径列表
        task_id: 任务ID
        
    Returns:
        图像分析任务
    """
    # 确保image_paths是列表格式
    if isinstance(image_paths, str):
        if os.path.isdir(image_paths):
            # 如果是目录，获取所有图像文件
            image_files = []
            for ext in ['.dcm', '.png', '.jpg', '.jpeg']:
                image_files.extend(
                    [os.path.join(image_paths, f) for f in os.listdir(image_paths) if f.lower().endswith(ext)]
                )
            image_paths = image_files
        else:
            # 如果是单个文件，转换为列表
            image_paths = [image_paths]
    
    # 构建任务描述
    image_count = len(image_paths)
    if image_count == 1:
        description = f"分析一张CT图像，识别其中的关键医学特征和可能的异常。"
    else:
        description = f"分析{image_count}张CT图像，识别其中的关键医学特征和可能的异常。"
    
    # 创建任务
    return Task(
        description=description,
        agent=agent,
        expected_output="""
        详细的CT图像分析报告，包括:
        1. 图像质量评估
        2. 正常解剖结构描述
        3. 异常发现的详细描述，包括位置、大小、形态、密度/强度等特征
        4. 关键医学特征的总结
        """,
        tools=[ct_analysis_tool.analyze_multiple_images],
        context=[
            f"需要分析的CT图像数量: {image_count}",
            f"图像路径: {', '.join(image_paths[:3])}{'...' if image_count > 3 else ''}",
        ],
        id=task_id
    )


def create_knowledge_retrieval_task(
    agent: Agent,
    retrieval_tool: MedicalKnowledgeRetrievalTool,
    ct_analysis_result: Dict[str, Any],
    task_id: str = "knowledge_retrieval"
) -> Task:
    """
    创建医学知识检索任务
    
    Args:
        agent: 分配任务的智能体
        retrieval_tool: 知识检索工具
        ct_analysis_result: CT分析结果
        task_id: 任务ID
        
    Returns:
        知识检索任务
    """
    # 提取CT分析描述
    if "summary" in ct_analysis_result:
        ct_description = ct_analysis_result["summary"].get("combined_description", "")
    elif "analysis" in ct_analysis_result:
        ct_description = ct_analysis_result["analysis"].get("combined_description", "")
    else:
        ct_description = str(ct_analysis_result)
    
    return Task(
        description="基于CT分析结果检索相关的医学知识和研究信息",
        agent=agent,
        expected_output="""
        全面的医学知识检索报告，包括:
        1. 与CT发现相关的疾病或病理说明
        2. 相关的诊断标准和鉴别诊断
        3. 最新的治疗指南和研究进展
        4. 预后和风险因素分析
        5. 临床建议的科学依据
        """,
        tools=[retrieval_tool.retrieve_knowledge_from_ct_analysis],
        context=[
            f"CT分析结果: {ct_description[:500]}{'...' if len(ct_description) > 500 else ''}",
            "检索相关的医学知识，重点关注与CT发现相符的疾病、症状、诊断标准和治疗方法"
        ],
        id=task_id
    )


def create_report_generation_task(
    agent: Agent,
    report_tool: MedicalReportGenerator,
    ct_analysis_result: Dict[str, Any],
    medical_knowledge: str,
    task_id: str = "report_generation"
) -> Task:
    """
    创建诊断报告生成任务
    
    Args:
        agent: 分配任务的智能体
        report_tool: 报告生成工具
        ct_analysis_result: CT分析结果
        medical_knowledge: 医学知识内容
        task_id: 任务ID
        
    Returns:
        报告生成任务
    """
    # 提取CT分析描述
    if "summary" in ct_analysis_result:
        ct_description = ct_analysis_result["summary"].get("combined_description", "")
        abnormality_detected = ct_analysis_result["summary"].get("abnormality_detected", False)
    elif "analysis" in ct_analysis_result:
        ct_description = ct_analysis_result["analysis"].get("combined_description", "")
        abnormality_detected = ct_analysis_result["analysis"].get("abnormality_detected", False)
    else:
        ct_description = str(ct_analysis_result)
        abnormality_detected = "异常" in ct_description
    
    # 根据是否发现异常调整任务描述
    if abnormality_detected:
        task_description = "生成详细的医学CT诊断报告，重点分析发现的异常"
    else:
        task_description = "生成医学CT诊断报告，确认正常发现并提供适当的建议"
    
    return Task(
        description=task_description,
        agent=agent,
        expected_output="""
        专业的CT诊断报告，包括:
        1. 详细的影像发现描述
        2. 专业的分析与解释
        3. 明确的诊断意见或鉴别诊断
        4. 具体的后续建议和处理方案
        """,
        tools=[report_tool.generate_ct_report],
        context=[
            f"CT分析结果: {ct_description[:300]}{'...' if len(ct_description) > 300 else ''}",
            f"医学知识参考: {medical_knowledge[:300]}{'...' if len(medical_knowledge) > 300 else ''}",
            "生成一份专业、全面且结构清晰的医学诊断报告"
        ],
        id=task_id
    )


def create_comparative_report_task(
    agent: Agent,
    report_tool: MedicalReportGenerator,
    current_ct_result: Dict[str, Any],
    previous_ct_result: Dict[str, Any],
    medical_knowledge: str,
    task_id: str = "comparative_report"
) -> Task:
    """
    创建对比报告生成任务
    
    Args:
        agent: 分配任务的智能体
        report_tool: 报告生成工具
        current_ct_result: 当前CT分析结果
        previous_ct_result: 之前的CT分析结果
        medical_knowledge: 医学知识内容
        task_id: 任务ID
        
    Returns:
        对比报告生成任务
    """
    # 提取CT分析描述
    if "summary" in current_ct_result:
        current_description = current_ct_result["summary"].get("combined_description", "")
    elif "analysis" in current_ct_result:
        current_description = current_ct_result["analysis"].get("combined_description", "")
    else:
        current_description = str(current_ct_result)
    
    if "summary" in previous_ct_result:
        previous_description = previous_ct_result["summary"].get("combined_description", "")
    elif "analysis" in previous_ct_result:
        previous_description = previous_ct_result["analysis"].get("combined_description", "")
    else:
        previous_description = str(previous_ct_result)
    
    return Task(
        description="比较当前与之前的CT扫描结果，生成对比分析报告",
        agent=agent,
        expected_output="""
        详细的CT对比分析报告，包括:
        1. 当前与之前CT的对比发现
        2. 变化的分析与解释
        3. 基于对比结果的诊断意见
        4. 针对疾病进展或改善的建议
        """,
        tools=[report_tool.generate_comparative_report],
        context=[
            f"当前CT分析: {current_description[:200]}...",
            f"之前CT分析: {previous_description[:200]}...",
            f"医学知识参考: {medical_knowledge[:200]}...",
            "重点分析两次CT扫描之间的变化和临床意义"
        ],
        id=task_id
    )
