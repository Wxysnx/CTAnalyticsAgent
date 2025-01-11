 """
测试Agent工具: 验证CT分析、知识检索和报告生成工具
"""
import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from tools.ct_analysis import BiomedCLIPTool, create_ct_analysis_tool
from tools.knowledge_retrieval import MedicalKnowledgeRetrievalTool
from tools.report_generation import MedicalReportGenerator
from utils.image_processing import preprocess_for_biomedclip


class TestCTAnalysisTool(unittest.TestCase):
    """测试CT分析工具"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建临时测试图像
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_image_path = os.path.join(cls.temp_dir.name, "test_image.png")
        
        # 创建简单的灰度图像
        img = Image.new('L', (224, 224), 128)
        img.save(cls.test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        cls.temp_dir.cleanup()
    
    @patch("tools.ct_analysis.AutoProcessor")
    @patch("tools.ct_analysis.AutoModel")
    def test_biomedclip_tool_init(self, mock_model, mock_processor):
        """测试BiomedCLIPTool初始化"""
        # 配置模拟对象
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        # 创建工具实例
        tool = BiomedCLIPTool()
        
        # 验证初始化行为
        self.assertIsNotNone(tool.processor)
        self.assertIsNotNone(tool.model)
        mock_processor.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
    
    @patch("tools.ct_analysis.BiomedCLIPTool._run_biomedclip_analysis")
    def test_analyze_regular_image(self, mock_run_analysis):
        """测试分析常规图像"""
        # 配置模拟对象
        mock_run_analysis.return_value = {
            "combined_description": "正常CT图像，无明显异常。",
            "abnormality_detected": False,
            "confidence": 0.95
        }
        
        # 创建带有模拟方法的工具实例
        tool = MagicMock(spec=BiomedCLIPTool)
        tool._analyze_regular_image = BiomedCLIPTool._analyze_regular_image
        tool._run_biomedclip_analysis = mock_run_analysis
        
        # 使用真实图像路径调用方法
        result = tool._analyze_regular_image(self, self.test_image_path)
        
        # 验证结果
        self.assertIn("metadata", result)
        self.assertIn("analysis", result)
        self.assertEqual(result["image_type"], "Regular")
        mock_run_analysis.assert_called_once()
    
    def test_create_ct_analysis_tool(self):
        """测试创建CT分析工具函数"""
        with patch("tools.ct_analysis.BiomedCLIPTool") as mock_tool_class:
            # 配置模拟对象
            mock_tool_class.return_value = MagicMock()
            
            # 调用函数
            tool = create_ct_analysis_tool()
            
            # 验证结果
            self.assertIsNotNone(tool)
            mock_tool_class.assert_called_once()


class TestKnowledgeRetrievalTool(unittest.TestCase):
    """测试知识检索工具"""

    def setUp(self):
        """设置测试环境"""
        # 模拟语言模型
        self.mock_llm = MagicMock()
        
        # 模拟检索器
        self.mock_retriever = MagicMock()
        self.mock_retriever.get_relevant_documents.return_value = [
            MagicMock(page_content="肺炎的CT表现", metadata={"source": "医学教科书"}),
            MagicMock(page_content="肺部结节的分类", metadata={"source": "放射学指南"})
        ]
    
    @patch("tools.knowledge_retrieval.get_medical_knowledge_retriever")
    def test_init_with_retriever(self, mock_get_retriever):
        """测试使用现有检索器初始化"""
        # 创建工具实例
        tool = MedicalKnowledgeRetrievalTool(
            llm=self.mock_llm,
            retriever=self.mock_retriever
        )
        
        # 验证初始化行为
        self.assertEqual(tool.llm, self.mock_llm)
        self.assertEqual(tool.retriever, self.mock_retriever)
        mock_get_retriever.assert_not_called()
    
    def test_retrieve_knowledge(self):
        """测试检索知识"""
        # 创建工具实例
        tool = MedicalKnowledgeRetrievalTool(
            llm=self.mock_llm,
            retriever=self.mock_retriever
        )
        
        # 调用检索方法
        result = tool.retrieve_knowledge("肺炎的CT表现")
        
        # 验证结果
        self.assertTrue(isinstance(result, str))
        self.mock_retriever.get_relevant_documents.assert_called_once()
    
    def test_retrieve_knowledge_return_documents(self):
        """测试检索知识并返回文档"""
        # 创建工具实例
        tool = MedicalKnowledgeRetrievalTool(
            llm=self.mock_llm,
            retriever=self.mock_retriever
        )
        
        # 调用检索方法
        result = tool.retrieve_knowledge("肺炎的CT表现", return_documents=True)
        
        # 验证结果
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 2)
        self.mock_retriever.get_relevant_documents.assert_called_once()
    
    @patch("tools.knowledge_retrieval.generate_multiple_queries")
    def test_retrieve_knowledge_from_ct_analysis(self, mock_generate_queries):
        """测试基于CT分析结果检索知识"""
        # 配置模拟对象
        mock_generate_queries.return_value = ["肺炎的CT表现", "肺部结节的特征"]
        
        # 创建工具实例
        tool = MedicalKnowledgeRetrievalTool(
            llm=self.mock_llm,
            retriever=self.mock_retriever
        )
        
        # 创建测试CT分析结果
        ct_result = {
            "summary": {
                "combined_description": "肺部有磨玻璃样阴影，考虑肺炎可能。"
            }
        }
        
        # 调用检索方法
        result = tool.retrieve_knowledge_from_ct_analysis(ct_result)
        
        # 验证结果
        self.assertTrue(isinstance(result, str))
        mock_generate_queries.assert_called_once()
        # 检索器应该被调用两次，对应两个生成的查询
        self.assertEqual(self.mock_retriever.get_relevant_documents.call_count, 2)


class TestReportGenerationTool(unittest.TestCase):
    """测试报告生成工具"""

    def setUp(self):
        """设置测试环境"""
        # 模拟语言模型
        self.mock_llm = MagicMock()
        self.mock_llm_chain = MagicMock()
        self.mock_llm_chain.run.return_value = """
        1. 影像发现: 肺部可见磨玻璃样阴影，分布于双肺下叶。
        
        2. 分析与解释: 磨玻璃样阴影通常提示间质性改变，考虑可能是炎症或早期间质性肺病。
        
        3. 诊断意见: 考虑为肺炎，也不排除间质性肺病的可能。
        
        4. 建议: 建议短期复查CT，进行血常规、C反应蛋白等炎症指标检查。
        """
        
        # 创建工具实例
        self.tool = MedicalReportGenerator(llm=self.mock_llm)
    
    @patch("tools.report_generation.LLMChain")
    def test_generate_ct_report(self, mock_llm_chain_class):
        """测试生成CT报告"""
        # 配置模拟对象
        mock_llm_chain_class.return_value = self.mock_llm_chain
        
        # 创建测试CT分析结果和医学知识
        ct_result = {
            "summary": {
                "combined_description": "肺部有磨玻璃样阴影，考虑肺炎可能。",
                "abnormality_detected": True,
                "scan_region": "胸部"
            }
        }
        medical_knowledge = "肺炎通常表现为磨玻璃样阴影。"
        
        # 调用生成报告方法
        result = self.tool.generate_ct_report(ct_result, medical_knowledge)
        
        # 验证结果
        self.assertIn("content", result)
        self.assertEqual(result["format"], "markdown")
        mock_llm_chain_class.assert_called_once()
        self.mock_llm_chain.run.assert_called_once()
    
    def test_extract_report_sections(self):
        """测试提取报告各个部分"""
        # 准备测试报告文本
        report_text = """
        1. 影像发现: 肺部可见磨玻璃样阴影，分布于双肺下叶。
        
        2. 分析与解释: 磨玻璃样阴影通常提示间质性改变，考虑可能是炎症或早期间质性肺病。
        
        3. 诊断意见: 考虑为肺炎，也不排除间质性肺病的可能。
        
        4. 建议: 建议短期复查CT，进行血常规、C反应蛋白等炎症指标检查。
        """
        
        # 调用提取方法
        sections = self.tool._extract_report_sections(report_text)
        
        # 验证结果
        self.assertIn("影像发现", sections)
        self.assertIn("分析与解释", sections)
        self.assertIn("诊断意见", sections)
        self.assertIn("建议", sections)
        self.assertTrue("肺部可见磨玻璃样阴影" in sections["影像发现"])
        self.assertTrue("肺炎" in sections["诊断意见"])
    
    @patch("tools.report_generation.save_report_to_markdown")
    def test_save_report_markdown(self, mock_save):
        """测试保存报告为Markdown"""
        # 配置模拟对象
        mock_save.return_value = "/tmp/report.md"
        
        # 准备测试报告数据
        report_data = {
            "content": "# 医学CT影像诊断报告\n\n## 影像发现\n肺部可见磨玻璃样阴影",
            "format": "markdown"
        }
        
        # 调用保存方法
        result = self.tool.save_report(report_data, "/tmp/report.md")
        
        # 验证结果
        self.assertEqual(result, "/tmp/report.md")
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
