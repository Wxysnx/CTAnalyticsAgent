 """
集成测试: 验证医学CT智能体系统的端到端功能
"""
import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import json
from PIL import Image

from langchain.schema.language_model import BaseLanguageModel
from crewai import Agent, Task, Crew

from tools.ct_analysis import BiomedCLIPTool, create_ct_analysis_tool
from tools.knowledge_retrieval import MedicalKnowledgeRetrievalTool
from tools.report_generation import MedicalReportGenerator
from crew.process import MedicalCTCrew


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建临时测试目录和图像
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_image_path = os.path.join(cls.temp_dir.name, "test_image.png")
        cls.output_path = os.path.join(cls.temp_dir.name, "report.md")
        
        # 创建简单的灰度图像
        img = Image.new('L', (224, 224), 128)
        img.save(cls.test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        cls.temp_dir.cleanup()

    @patch("crew.process.create_medical_ct_agents")
    @patch("tools.ct_analysis.BiomedCLIPTool")
    @patch("tools.knowledge_retrieval.MedicalKnowledgeRetrievalTool")
    @patch("tools.report_generation.MedicalReportGenerator")
    @patch("crew.process.Crew")
    def test_medical_ct_crew_analyze(
        self, 
        mock_crew_class, 
        mock_report_tool_class, 
        mock_knowledge_tool_class, 
        mock_ct_tool_class, 
        mock_create_agents
    ):
        """测试医学CT智能体团队的分析功能"""
        # 配置模拟对象
        mock_llm = MagicMock(spec=BaseLanguageModel)
        
        mock_ct_tool = MagicMock()
        mock_ct_tool_class.return_value = mock_ct_tool
        
        mock_knowledge_tool = MagicMock()
        mock_knowledge_tool_class.return_value = mock_knowledge_tool
        
        mock_report_tool = MagicMock()
        mock_report_tool_class.return_value = mock_report_tool
        
        mock_agents = {
            "image_analyst": MagicMock(spec=Agent),
            "medical_researcher": MagicMock(spec=Agent),
            "radiologist": MagicMock(spec=Agent)
        }
        mock_create_agents.return_value = mock_agents
        
        mock_crew = MagicMock(spec=Crew)
        mock_crew_class.return_value = mock_crew
        
        # 模拟智能体团队的执行结果
        mock_crew.kickoff.side_effect = [
            # 第一次调用返回CT分析结果
            '''{"analysis": {"combined_description": "肺部有磨玻璃样阴影，考虑肺炎可能。", "abnormality_detected": true}}''',
            # 第二次调用返回医学知识
            "肺炎通常表现为磨玻璃样阴影，可能是由细菌或病毒感染引起。",
            # 第三次调用返回报告结果
            '''{"content": "# 医学CT影像诊断报告\\n\\n## 影像发现\\n肺部可见磨玻璃样阴影", "format": "markdown"}'''
        ]
        
        # 创建医学CT智能体团队
        crew = MedicalCTCrew(
            llm=mock_llm,
            ct_analysis_tool=mock_ct_tool,
            knowledge_tool=mock_knowledge_tool,
            report_tool=mock_report_tool
        )
        
        # 执行分析
        result = crew.analyze_ct_images(
            image_paths=self.test_image_path,
            output_path=self.output_path
        )
        
        # 验证结果
        self.assertIn("ct_analysis", result)
        self.assertIn("medical_knowledge", result)
        self.assertIn("report", result)
        
        # 验证Crew被调用了3次
        self.assertEqual(mock_crew.kickoff.call_count, 3)
        
        # 验证报告保存功能被调用
        mock_report_tool.save_report.assert_called_once()
    
    @patch("tools.ct_analysis.BiomedCLIPTool.analyze_multiple_images")
    @patch("tools.knowledge_retrieval.MedicalKnowledgeRetrievalTool.retrieve_knowledge")
    @patch("crew.process.create_medical_ct_agents")
    @patch("crew.process.Crew")
    def test_compare_ct_scans(
        self, 
        mock_crew_class, 
        mock_create_agents, 
        mock_retrieve_knowledge, 
        mock_analyze_images
    ):
        """测试比较CT扫描功能"""
        # 配置模拟对象
        mock_llm = MagicMock(spec=BaseLanguageModel)
        
        mock_analyze_images.side_effect = [
            # 第一次调用返回当前CT分析结果
            {"summary": {"combined_description": "肺部有磨玻璃样阴影，考虑肺炎可能。"}},
            # 第二次调用返回之前的CT分析结果
            {"summary": {"combined_description": "肺部无明显异常。"}}
        ]
        
        mock_retrieve_knowledge.return_value = "肺炎通常表现为磨玻璃样阴影，可能是由细菌或病毒感染引起。"
        
        mock_agents = {
            "radiologist": MagicMock(spec=Agent)
        }
        mock_create_agents.return_value = mock_agents
        
        mock_crew = MagicMock(spec=Crew)
        mock_crew_class.return_value = mock_crew
        mock_crew.kickoff.return_value = '''{"content": "# CT对比分析报告\\n\\n## 影像对比发现\\n当前扫描显示肺部有新出现的磨玻璃样阴影。", "format": "markdown"}'''
        
        # 创建医学CT智能体团队
        ct_tool = BiomedCLIPTool()  # 使用真实的类，但方法会被模拟
        knowledge_tool = MagicMock(spec=MedicalKnowledgeRetrievalTool)
        knowledge_tool.retrieve_knowledge = mock_retrieve_knowledge
        report_tool = MagicMock(spec=MedicalReportGenerator)
        
        crew = MedicalCTCrew(
            llm=mock_llm,
            ct_analysis_tool=ct_tool,
            knowledge_tool=knowledge_tool,
            report_tool=report_tool
        )
        
        # 执行对比分析
        result = crew.compare_ct_scans(
            current_image_paths=self.test_image_path,
            previous_image_paths=self.test_image_path,
            output_path=self.output_path
        )
        
        # 验证结果
        self.assertIn("current_ct_analysis", result)
        self.assertIn("previous_ct_analysis", result)
        self.assertIn("medical_knowledge", result)
        self.assertIn("comparative_report", result)
        
        # 验证方法调用
        self.assertEqual(mock_analyze_images.call_count, 2)
        mock_retrieve_knowledge.assert_called_once()
        mock_crew.kickoff.assert_called_once()
        report_tool.save_report.assert_called_once()


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 跳过实际执行的测试，除非设置了特定环境变量
        if not os.environ.get("RUN_E2E_TESTS"):
            self.skipTest("跳过端到端测试。设置 RUN_E2E_TESTS=1 环境变量以启用。")
        
        # 创建临时测试目录
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # 检查是否提供了测试图像路径
        self.test_image_path = os.environ.get("TEST_CT_IMAGE")
        if not self.test_image_path or not os.path.exists(self.test_image_path):
            self.skipTest("未提供有效的测试CT图像路径。设置 TEST_CT_IMAGE 环境变量。")
    
    def tearDown(self):
        """清理测试环境"""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
    
    @unittest.skip("此测试需要完整环境和API密钥，默认跳过")
    def test_full_system(self):
        """完整系统测试"""
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from main import setup_llm, setup_tools, analyze_ct_images
        
        # 设置语言模型
        llm = setup_llm("openai")
        
        # 设置工具
        tools = setup_tools(llm)
        
        # 执行分析
        output_dir = self.temp_dir.name
        result = analyze_ct_images(
            image_paths=self.test_image_path,
            tools=tools,
            llm=llm,
            output_dir=output_dir
        )
        
        # 验证结果
        self.assertIn("ct_analysis", result)
        self.assertIn("medical_knowledge", result)
        self.assertIn("report", result)
        
        # 验证生成了报告文件
        report_files = [f for f in os.listdir(output_dir) if f.endswith('.md')]
        self.assertGreater(len(report_files), 0)
        
        # 检查报告内容
        with open(os.path.join(output_dir, report_files[0]), 'r', encoding='utf-8') as f:
            report_content = f.read()
            self.assertIn("影像发现", report_content)
            self.assertIn("诊断意见", report_content)


if __name__ == "__main__":
    unittest.main()
