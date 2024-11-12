 """
配置文件: 包含项目的所有配置参数
"""
import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MEDICAL_DOCS_DIR = DATA_DIR / "medical_docs"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 确保所需目录存在
os.makedirs(MEDICAL_DOCS_DIR, exist_ok=True)
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# 模型配置
BIOMEDCLIP_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_MODEL_NAME = "medalpaca/medalpaca-7b"  # 示例，实际使用时可能需要调整

# LangChain配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVALS = 5
SIMILARITY_THRESHOLD = 0.75

# DICOM处理配置
DEFAULT_WINDOW_CENTER = 50
DEFAULT_WINDOW_WIDTH = 400

# 创建日志目录
LOG_DIR = BASE_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# API密钥配置 (请在实际部署中使用环境变量或安全存储)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# CrewAI配置
# 如果需要使用OpenAI模型，修改为对应模型名称
CREWAI_LLM_MODEL = "gpt-4"
CREWAI_VERBOSE = True

# 报告生成配置
REPORT_TEMPLATE = """
# 医学CT影像诊断报告

## 患者信息
- **检查日期**: {examination_date}
- **检查类型**: CT扫描
- **检查部位**: {examination_area}

## 影像发现
{image_findings}

## 分析与解释
{analysis_and_interpretation}

## 诊断意见
{diagnostic_opinion}

## 建议
{recommendations}

## 报告日期
{report_date}

"""
