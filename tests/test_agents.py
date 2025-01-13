 """
测试CrewAI智能体: 验证医学CT分析智能体的创建和功能
"""
import unittest
from unittest.mock import patch, MagicMock

from langchain.schema.language_model import BaseLanguageModel
from crewai import Agent

from crew.agents import (
    get_image_analyst_agent,
    get_medical_researcher_agent,
    get_radiologist_agent,
    create_medical_ct_agents
)


class TestAgents(unittest.TestCase):
    """测试CrewAI智能体"""

    def setUp(self):
        """设置测试环境"""
        # 创建模拟的语言模型
        self.mock_llm = MagicMock(spec=BaseLanguageModel)
    
    def test_get_image_analyst_agent(self):
        """测试创建医学影像分析师智能体"""
        # 创建智能体
        agent = get_image_analyst_agent(llm=self.mock_llm)
        
        # 验证智能体属性
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.role, "医学影像分析师")
        self.assertTrue("分析CT图像" in agent.goal)
        self.assertTrue(agent.verbose)
        self.assertTrue(agent.allow_delegation)
    
    def test_get_medical_researcher_agent(self):
        """测试创建医学研究员智能体"""
        # 创建智能体
        agent = get_medical_researcher_agent(llm=self.mock_llm)
        
        # 验证智能体属性
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.role, "医学研究员")
        self.assertTrue("检索" in agent.goal)
        self.assertTrue(agent.verbose)
        self.assertTrue(agent.allow_delegation)
    
    def test_get_radiologist_agent(self):
        """测试创建放射科医师智能体"""
        # 创建智能体
        agent = get_radiologist_agent(llm=self.mock_llm)
        
        # 验证智能体属性
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.role, "放射科医师")
        self.assertTrue("诊断报告" in agent.goal)
        self.assertTrue(agent.verbose)
        self.assertTrue(agent.allow_delegation)
    
    def test_create_medical_ct_agents(self):
        """测试创建所有医学CT分析智能体"""
        # 创建所有智能体
        agents = create_medical_ct_agents(llm=self.mock_llm)
        
        # 验证返回的字典
        self.assertIsInstance(agents, dict)
        self.assertIn("image_analyst", agents)
        self.assertIn("medical_researcher", agents)
        self.assertIn("radiologist", agents)
        
        # 验证每个智能体的类型
        for name, agent in agents.items():
            self.assertIsInstance(agent, Agent)
    
    def test_agent_delegation_config(self):
        """测试智能体委派配置"""
        # 测试不允许委派
        agent = get_image_analyst_agent(llm=self.mock_llm, allow_delegation=False)
        self.assertFalse(agent.allow_delegation)
        
        # 测试允许委派
        agent = get_image_analyst_agent(llm=self.mock_llm, allow_delegation=True)
        self.assertTrue(agent.allow_delegation)


if __name__ == "__main__":
    unittest.main()
