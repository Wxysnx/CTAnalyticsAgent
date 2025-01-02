 """
定义医学CT分析的智能体: 影像分析师、医学研究员、放射科医师
"""
from typing import Dict, Any, List, Optional
import os

from crewai import Agent
from langchain_core.language_models import BaseLanguageModel

from config import CREWAI_LLM_MODEL


def get_image_analyst_agent(
    llm: Optional[BaseLanguageModel] = None,
    allow_delegation: bool = True
) -> Agent:
    """
    创建医学影像分析师智能体
    
    Args:
        llm: 语言模型
        allow_delegation: 是否允许委派任务
        
    Returns:
        医学影像分析师智能体
    """
    return Agent(
        role="医学影像分析师",
        goal="精确分析CT图像，识别关键特征并提供专业描述",
        backstory="""
        你是一位经验丰富的医学影像分析师，拥有医学影像学博士学位和十年临床经验。
        你擅长使用先进的AI模型分析CT图像，能够发现细微的异常特征。
        你的专长是结合图像特征与解剖学知识，提供准确的初步分析。
        你对各种病理表现的CT影像特征有深入了解，能够识别微妙的变化和异常模式。
        """,
        verbose=True,
        llm=llm,
        allow_delegation=allow_delegation
    )


def get_medical_researcher_agent(
    llm: Optional[BaseLanguageModel] = None,
    allow_delegation: bool = True
) -> Agent:
    """
    创建医学研究员智能体
    
    Args:
        llm: 语言模型
        allow_delegation: 是否允许委派任务
        
    Returns:
        医学研究员智能体
    """
    return Agent(
        role="医学研究员",
        goal="检索和分析与CT发现相关的最新医学知识，提供科学依据",
        backstory="""
        你是一位医学研究员，拥有医学博士学位和流行病学硕士学位。
        你擅长检索和解读最新的医学研究文献，能够将复杂的医学概念转化为清晰的解释。
        你对循证医学有深入理解，能够评估证据的强度和适用性。
        你熟悉各种疾病的最新诊断标准、治疗指南和预后因素，能够提供全面的医学背景知识。
        你善于整合多源信息，为临床决策提供全面的知识支持。
        """,
        verbose=True,
        llm=llm,
        allow_delegation=allow_delegation
    )


def get_radiologist_agent(
    llm: Optional[BaseLanguageModel] = None,
    allow_delegation: bool = True
) -> Agent:
    """
    创建放射科医师智能体
    
    Args:
        llm: 语言模型
        allow_delegation: 是否允许委派任务
        
    Returns:
        放射科医师智能体
    """
    return Agent(
        role="放射科医师",
        goal="整合图像分析和医学知识，提供专业的诊断报告和治疗建议",
        backstory="""
        你是一位资深放射科医师，拥有放射诊断学副教授职称和15年临床经验。
        你在胸部和腹部CT诊断领域有特殊专长，曾在顶级医学期刊发表多篇研究论文。
        你善于整合影像发现与临床信息，提供全面准确的诊断和治疗建议。
        你具有出色的医学判断力，能够权衡不同诊断假设的可能性，并提供合理的鉴别诊断。
        你精通医学报告写作，能够撰写专业、清晰且符合临床需求的诊断报告。
        """,
        verbose=True,
        llm=llm,
        allow_delegation=allow_delegation
    )


def create_medical_ct_agents(
    llm: Optional[BaseLanguageModel] = None
) -> Dict[str, Agent]:
    """
    创建所有医学CT分析的智能体
    
    Args:
        llm: 语言模型，如果未提供将使用默认模型
        
    Returns:
        包含所有智能体的字典
    """
    agents = {
        "image_analyst": get_image_analyst_agent(llm=llm),
        "medical_researcher": get_medical_researcher_agent(llm=llm),
        "radiologist": get_radiologist_agent(llm=llm)
    }
    
    return agents
