# Medical CT Agent

![Medical CT Analysis](https://img.shields.io/badge/AI-Medical%20Imaging-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-0.28.0%2B-orange)
![LangChain](https://img.shields.io/badge/LangChain-0.0.267%2B-yellow)
![BiomedCLIP](https://img.shields.io/badge/BiomedCLIP-Vision%20Model-violet)
![RAG](https://img.shields.io/badge/Architecture-RAG-red)

## 项目概述

CTAnalyticsAgent 是一个先进的医学CT图像分析系统，使用多智能体协作框架和最新的AI技术自动分析CT图像、检索医学知识和生成专业诊断报告。该系统集成了计算机视觉、自然语言处理和知识检索技术，为放射科医生提供智能辅助诊断工具。

## 核心功能

- 🧠 **多智能体协作**：使用CrewAI构建专家智能体团队，包括影像分析师、医学研究员和放射科医师
- 🔍 **CT图像自动分析**：通过BiomedCLIP模型分析CT图像，识别关键特征和异常
- 📚 **医学知识检索**：基于RAG架构的医学知识库检索，提供相关医学依据
- 📝 **专业诊断报告生成**：自动生成结构化的医学诊断报告
- 📊 **历史扫描对比分析**：比较多次CT扫描，自动检测变化和疾病进展

## 技术架构

![Architecture](https://mermaid.ink/img/pako:eNqNVE1v2zAM_SuETgm6ediBw7IchhbdShTIoUCAYD0MBWKrsYHIUiTJSdqh_32U7I_YyTrsYlB8j48PKoDbkqBES82ZMtBYLKEGQ8XREskQHQX9mWBfMI67unCxYsU0pQVNBvhgM58P9lrVkAQS3U7eo3QvngR62GVpzaISB-v5W0O1wN6SC92G_wlntDT3mXaqzuIjcUUcwtOJtmVpUrJRLQ478VbKNvgixVDNZLzm1WWnNvuX4r2gTbRVlmspWFCKLBuMrpbRLnI8Pi4HWdbGtnGNOtVmogaxSG1fJfR4jus2GGb5YmEqt7XnruDR42ksUTBz_1eS3OKGWe_NihYVmxj0Ok6tZkzjFx9j2tmzNxLJuBtaBjNTU3VGreBv5QdYG7IKiLk1vKFNNQtVp3moRzwk4yg_p6UptKHnxaIdDLznaeK_U8kFWdkjvT4OrmZRj9YLbShBPQ3uo-mmG02lq5FluR9ibG494sGMeZ9G9ZNV7MxVtTfGPYnKRzNj-d7Vw82FwZW5qhqsg5eW7bcweSNNUR5hHQUA5V2xLTGswqBizfMBlt0Y7AxJnKqE1TdUR3gik3idSmOoBnJ-p8s5_tX3VruD_2dXwp6MCg0t6gNzT3XP_n5F_2JI9Md9dpBPwcvTCIlXI1QfISyvRqhOhUl9tXsK9me6ZBhdkfT-32ywJzwqboPyXDy4dSiRKtKEQzRA6z7nc1TOoPwe_ADwShbr)

### 前沿AI技术栈

- **CrewAI**：最新的多智能体框架，允许角色专业化智能体协作
- **LangChain**：组合LLM与应用的顶级框架，实现复杂AI工作流
- **BiomedCLIP**：微软专为医学图像分析优化的多模态AI模型
- **RAG架构**：检索增强生成技术，提供基于医学文献的可靠信息
- **ChromaDB**：高效向量数据库，支持语义搜索
- **DICOM处理**：专业医学影像格式解析与处理
- **OpenAI集成**：与GPT-4等高级模型集成，支持医学推理

### 系统组件

1. **CT影像分析模块**
   - BiomedCLIP视觉模型集成
   - 医学图像预处理流水线
   - 自适应窗宽窗位调整算法

2. **知识检索引擎**
   - 医学文档向量化与索引
   - 多查询生成策略
   - 相关文档重排序与提取

3. **多智能体系统**
   - 专业化角色智能体
   - 任务规划与协调
   - 结果整合与推理

4. **报告生成系统**
   - 结构化医学诊断模板
   - 历史对比分析
   - 临床建议生成


## 项目结构

```
medical-ct-agent/
├── main.py                      # 项目主入口，初始化和启动系统
├── config.py                    # 配置文件（API密钥、模型路径等）
├── crew/                        # CrewAI相关实现
│   ├── agents.py                # 定义专业Agent（影像分析师、医学研究员、放射科医师）
│   ├── tasks.py                 # 定义CrewAI任务
│   └── process.py               # 定义CrewAI工作流程
├── tools/                       # CrewAI Agent使用的工具
│   ├── ct_analysis.py           # CT影像分析工具（使用BiomedCLIP）
│   ├── knowledge_retrieval.py   # 知识检索工具（使用LangChain RAG）
│   └── report_generation.py     # 报告生成工具
├── langchain_components/        # LangChain组件配置
│   ├── document_loaders.py      # 配置文档加载器
│   ├── embeddings.py            # 配置嵌入模型
│   ├── vectorstore.py           # 配置向量存储
│   └── retriever.py             # 配置检索器
└── utils/                       # 辅助工具
    ├── image_processing.py      # 图像预处理功能
    ├── dicom_handler.py         # DICOM格式处理
    └── report_formatter.py      # 报告格式化
```

## 算法与技术亮点

- 🔄 **多模态融合**: 将图像数据与医学文本知识无缝结合
- 🔗 **可解释AI**: 智能体提供诊断推理过程，增强医生信任
- 🧮 **自适应窗宽处理**: 优化CT图像对比度，增强病理特征可见度
- 📈 **多查询生成**: 通过LLM分解复杂医学描述为多个精确查询
- 📌 **相似度重排序**: 使用余弦相似度重新排序检索结果，提高相关性
- 🏥 **专业报告格式化**: 符合医学标准的结构化报告生成

## 运行示例

```bash
# 基本使用
python main.py --image_path ./data/sample_images/chest_ct_001.dcm --output_dir ./reports

# 多图像分析
python main.py --image_path ./data/sample_images/ --model openai

# 使用本地模型
python main.py --image_path ./data/sample_images/ --model local
