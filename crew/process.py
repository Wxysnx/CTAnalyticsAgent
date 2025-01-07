 """
定义CrewAI工作流程: 协调智能体任务流程
"""
from typing import Dict, Any, List, Optional, Union
import os

from crewai import Crew, Process
from crewai.agent import Agent
from crewai.task import Task
from langchain_core.language_models import BaseLanguageModel

from config import CREWAI_VERBOSE
from tools.ct_analysis import BiomedCLIPTool, create_ct_analysis_tool
from tools.knowledge_retrieval import MedicalKnowledgeRetrievalTool
from tools.report_generation import MedicalReportGenerator
from crew.agents import create_medical_ct_agents
from crew.tasks import (
    create_image_analysis_task,
    create_knowledge_retrieval_task,
    create_report_generation_task,
    create_comparative_report_task
)


class MedicalCTCrew:
    """医学CT分析智能体团队"""
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        ct_analysis_tool: Optional[BiomedCLIPTool] = None,
        knowledge_tool: Optional[MedicalKnowledgeRetrievalTool] = None,
        report_tool: Optional[MedicalReportGenerator] = None
    ):
        """
        初始化医学CT分析智能体团队
        
        Args:
            llm: 语言模型
            ct_analysis_tool: CT分析工具
            knowledge_tool: 知识检索工具
            report_tool: 报告生成工具
        """
        # 创建智能体
        self.agents = create_medical_ct_agents(llm=llm)
        
        # 保存语言模型
        self.llm = llm
        
        # 保存或创建工具
        self.ct_analysis_tool = ct_analysis_tool or create_ct_analysis_tool()
        self.knowledge_tool = knowledge_tool
        self.report_tool = report_tool
        
        # 验证所需工具
        if not self.knowledge_tool:
            raise ValueError("需要提供知识检索工具")
        if not self.report_tool:
            raise ValueError("需要提供报告生成工具")
    
    def analyze_ct_images(
        self,
        image_paths: Union[str, List[str]],
        process_type: str = "sequential",
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析CT图像并生成诊断报告
        
        Args:
            image_paths: CT图像路径或路径列表
            process_type: 处理类型 ("sequential" 或 "hierarchical")
            output_path: 报告输出路径
            
        Returns:
            分析结果字典
        """
        # 1. 创建图像分析任务
        image_analysis_task = create_image_analysis_task(
            agent=self.agents["image_analyst"],
            ct_analysis_tool=self.ct_analysis_tool,
            image_paths=image_paths
        )
        
        # 2. 创建知识检索和报告生成任务（为后续步骤准备）
        # 这些任务需要在第一个任务完成后才能具体化参数，先定义为None
        knowledge_retrieval_task = None
        report_generation_task = None
        
        # 3. 创建智能体团队
        crew = Crew(
            agents=[
                self.agents["image_analyst"],
                self.agents["medical_researcher"],
                self.agents["radiologist"]
            ],
            tasks=[image_analysis_task],  # 先只添加第一个任务
            verbose=CREWAI_VERBOSE,
            process=Process.sequential if process_type == "sequential" else Process.hierarchical
        )
        
        # 4. 执行图像分析任务
        print("执行CT图像分析任务...")
        image_analysis_result = crew.kickoff()
        
        # 5. 解析图像分析结果
        try:
            ct_analysis_result = eval(image_analysis_result)
        except:
            # 如果无法解析为字典，则作为字符串处理
            ct_analysis_result = {"analysis": {"combined_description": image_analysis_result}}
        
        # 6. 创建知识检索任务并执行
        knowledge_retrieval_task = create_knowledge_retrieval_task(
            agent=self.agents["medical_researcher"],
            retrieval_tool=self.knowledge_tool,
            ct_analysis_result=ct_analysis_result
        )
        
        # 更新团队任务
        crew.tasks = [knowledge_retrieval_task]
        
        print("执行医学知识检索任务...")
        medical_knowledge = crew.kickoff()
        
        # 7. 创建报告生成任务并执行
        report_generation_task = create_report_generation_task(
            agent=self.agents["radiologist"],
            report_tool=self.report_tool,
            ct_analysis_result=ct_analysis_result,
            medical_knowledge=medical_knowledge
        )
        
        # 更新团队任务
        crew.tasks = [report_generation_task]
        
        print("执行诊断报告生成任务...")
        report_result = crew.kickoff()
        
        # 8. 解析报告结果
        try:
            report_data = eval(report_result)
        except:
            # 如果无法解析为字典，则作为字符串处理
            report_data = {"content": report_result, "format": "markdown"}
        
        # 9. 保存报告（如果指定了输出路径）
        if output_path:
            self.report_tool.save_report(report_data, output_path)
            print(f"报告已保存至: {output_path}")
        
        # 10. 返回完整结果
        return {
            "ct_analysis": ct_analysis_result,
            "medical_knowledge": medical_knowledge,
            "report": report_data
        }
    
    def compare_ct_scans(
        self,
        current_image_paths: Union[str, List[str]],
        previous_image_paths: Union[str, List[str]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        比较当前与之前的CT扫描结果
        
        Args:
            current_image_paths: 当前CT图像路径或路径列表
            previous_image_paths: 之前CT图像路径或路径列表
            output_path: 报告输出路径
            
        Returns:
            比较分析结果字典
        """
        # 1. 分析当前CT图像
        print("分析当前CT图像...")
        current_ct_result = self.ct_analysis_tool.analyze_multiple_images(
            image_paths=current_image_paths
        )
        
        # 2. 分析之前的CT图像
        print("分析之前CT图像...")
        previous_ct_result = self.ct_analysis_tool.analyze_multiple_images(
            image_paths=previous_image_paths
        )
        
        # 3. 基于两组CT分析结果检索医学知识
        print("检索相关医学知识...")
        # 结合两组CT分析的结果
        combined_description = ""
        if "summary" in current_ct_result:
            combined_description += current_ct_result["summary"].get("combined_description", "") + " "
        if "summary" in previous_ct_result:
            combined_description += previous_ct_result["summary"].get("combined_description", "")
        
        medical_knowledge = self.knowledge_tool.retrieve_knowledge(combined_description)
        
        # 4. 创建对比报告任务
        comparative_report_task = create_comparative_report_task(
            agent=self.agents["radiologist"],
            report_tool=self.report_tool,
            current_ct_result=current_ct_result,
            previous_ct_result=previous_ct_result,
            medical_knowledge=medical_knowledge
        )
        
        # 5. 创建智能体团队
        crew = Crew(
            agents=[self.agents["radiologist"]],
            tasks=[comparative_report_task],
            verbose=CREWAI_VERBOSE,
            process=Process.sequential
        )
        
        # 6. 执行对比报告任务
        print("生成对比分析报告...")
        comparative_report = crew.kickoff()
        
        # 7. 解析报告结果
        try:
            report_data = eval(comparative_report)
        except:
            # 如果无法解析为字典，则作为字符串处理
            report_data = {"content": comparative_report, "format": "markdown"}
        
        # 8. 保存报告（如果指定了输出路径）
        if output_path:
            self.report_tool.save_report(report_data, output_path)
            print(f"对比分析报告已保存至: {output_path}")
        
        # 9. 返回完整结果
        return {
            "current_ct_analysis": current_ct_result,
            "previous_ct_analysis": previous_ct_result,
            "medical_knowledge": medical_knowledge,
            "comparative_report": report_data
        }
