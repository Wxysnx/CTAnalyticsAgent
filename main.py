 """
医学CT智能体系统入口: 初始化和启动系统
"""
import os
import argparse
from typing import Dict, Any, List, Optional, Union
import datetime

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

from tools.ct_analysis import create_ct_analysis_tool
from tools.knowledge_retrieval import MedicalKnowledgeRetrievalTool
from tools.report_generation import MedicalReportGenerator
from crew.process import MedicalCTCrew
from config import (
    OPENAI_API_KEY, 
    CREWAI_LLM_MODEL,
    DATA_DIR,
    SAMPLE_IMAGES_DIR,
    LOG_DIR
)


def setup_llm(model_type: str = "openai") -> Any:
    """
    设置语言模型
    
    Args:
        model_type: 模型类型 ("openai" 或 "local")
        
    Returns:
        语言模型实例
    """
    if model_type == "openai":
        # 使用OpenAI模型
        if not OPENAI_API_KEY:
            raise ValueError("使用OpenAI需要设置OPENAI_API_KEY环境变量")
        
        return ChatOpenAI(
            model=CREWAI_LLM_MODEL,
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY
        )
    else:
        # 使用本地模型
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            # 指定本地医学模型
            model_path = "medalpaca/medalpaca-7b"  # 可以替换为其他适合医学的本地模型路径
            
            print(f"加载本地模型: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # 创建pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            # 创建LangChain包装器
            return HuggingFacePipeline(pipeline=pipe)
            
        except ImportError:
            print("未安装transformers或torch，回退到使用OpenAI API")
            return ChatOpenAI(
                model=CREWAI_LLM_MODEL,
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY
            )


def get_sample_images() -> List[str]:
    """
    获取示例图像路径列表
    
    Returns:
        图像路径列表
    """
    if not os.path.exists(SAMPLE_IMAGES_DIR):
        os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
        print(f"示例图像目录已创建: {SAMPLE_IMAGES_DIR}")
        print("请在此目录中添加CT图像文件")
        return []
    
    # 获取所有图像文件
    image_files = []
    for ext in ['.dcm', '.png', '.jpg', '.jpeg']:
        image_files.extend(
            [os.path.join(SAMPLE_IMAGES_DIR, f) for f in os.listdir(SAMPLE_IMAGES_DIR) if f.lower().endswith(ext)]
        )
    
    return image_files


def setup_tools(llm: Any) -> Dict[str, Any]:
    """
    设置所需工具
    
    Args:
        llm: 语言模型
        
    Returns:
        工具字典
    """
    # 创建CT分析工具
    ct_analysis_tool = create_ct_analysis_tool()
    
    # 创建知识检索工具
    knowledge_tool = MedicalKnowledgeRetrievalTool(llm=llm)
    
    # 创建报告生成工具
    report_tool = MedicalReportGenerator(llm=llm)
    
    return {
        "ct_analysis_tool": ct_analysis_tool,
        "knowledge_tool": knowledge_tool,
        "report_tool": report_tool
    }


def analyze_ct_images(
    image_paths: Union[str, List[str]],
    tools: Dict[str, Any],
    llm: Any,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    分析CT图像并生成诊断报告
    
    Args:
        image_paths: 图像路径或路径列表
        tools: 工具字典
        llm: 语言模型
        output_dir: 输出目录
        
    Returns:
        分析结果
    """
    # 创建医学CT智能体团队
    crew = MedicalCTCrew(
        llm=llm,
        ct_analysis_tool=tools["ct_analysis_tool"],
        knowledge_tool=tools["knowledge_tool"],
        report_tool=tools["report_tool"]
    )
    
    # 确定输出路径
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"ct_report_{timestamp}.md")
    else:
        output_path = None
    
    # 执行分析
    result = crew.analyze_ct_images(
        image_paths=image_paths,
        output_path=output_path
    )
    
    return result


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="医学CT智能体系统")
    parser.add_argument("--image_path", type=str, help="CT图像文件或目录路径")
    parser.add_argument("--model", type=str, default="openai", choices=["openai", "local"], help="使用的语言模型类型")
    parser.add_argument("--output_dir", type=str, default=os.path.join(LOG_DIR, "reports"), help="输出目录")
    args = parser.parse_args()
    
    # 设置语言模型
    print("初始化语言模型...")
    llm = setup_llm(args.model)
    
    # 设置工具
    print("初始化工具...")
    tools = setup_tools(llm)
    
    # 确定分析的图像
    if args.image_path:
        image_paths = args.image_path
        if os.path.isdir(args.image_path):
            # 如果提供的是目录，获取所有图像文件
            image_files = []
            for ext in ['.dcm', '.png', '.jpg', '.jpeg']:
                image_files.extend(
                    [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) if f.lower().endswith(ext)]
                )
            image_paths = image_files
    else:
        # 使用示例图像
        image_paths = get_sample_images()
        if not image_paths:
            print("未找到任何图像文件，请提供有效的图像路径")
            return
    
    # 执行CT图像分析
    print(f"开始分析CT图像: {image_paths if isinstance(image_paths, str) else len(image_paths)}张图像")
    result = analyze_ct_images(
        image_paths=image_paths,
        tools=tools,
        llm=llm,
        output_dir=args.output_dir
    )
    
    # 输出结果摘要
    report = result.get("report", {})
    if isinstance(report, dict) and "content" in report:
        print("\n==== 诊断报告摘要 ====")
        content = report["content"]
        # 打印前500个字符
        print(f"{content[:500]}...")
        print("\n完整报告已保存到输出目录")
    else:
        print("\n==== 诊断报告 ====")
        print(report)


if __name__ == "__main__":
    main()
