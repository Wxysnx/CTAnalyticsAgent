 """
嵌入模型模块: 提供医学文本嵌入功能
"""
import os
from typing import List, Dict, Any, Optional, Union

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL_NAME, OPENAI_API_KEY


def get_medical_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    use_openai: bool = False,
    device: str = "cpu"
) -> Embeddings:
    """
    获取医学文本嵌入模型
    
    Args:
        model_name: 模型名称，默认使用配置中设置的医学嵌入模型
        use_openai: 是否使用OpenAI嵌入模型，需要API密钥
        device: 模型运行设备 ("cpu" 或 "cuda")
        
    Returns:
        嵌入模型实例
    """
    if use_openai and OPENAI_API_KEY:
        # 如果选择使用OpenAI的嵌入模型
        return OpenAIEmbeddings(
            model="text-embedding-3-small" if model_name == "default" else model_name,
            openai_api_key=OPENAI_API_KEY
        )
    else:
        # 使用本地Hugging Face模型
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )


def get_pubmedbert_embedding_model(device: str = "cpu") -> Embeddings:
    """
    获取专门优化用于医学文本的S-PubMedBert嵌入模型
    
    Args:
        device: 模型运行设备 ("cpu" 或 "cuda")
        
    Returns:
        PubMedBERT嵌入模型实例
    """
    return get_medical_embedding_model(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO",
        device=device
    )


def get_biomedclip_embedding_model(device: str = "cpu") -> Embeddings:
    """
    获取BiomedCLIP的文本嵌入模型部分
    
    Args:
        device: 模型运行设备 ("cpu" 或 "cuda")
        
    Returns:
        BiomedCLIP文本嵌入模型实例
    """
    return get_medical_embedding_model(
        model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device=device
    )


def compare_embeddings(text1: str, text2: str, embedding_model: Optional[Embeddings] = None) -> float:
    """
    计算两段文本的嵌入相似度
    
    Args:
        text1: 第一段文本
        text2: 第二段文本
        embedding_model: 嵌入模型，如果未提供则使用默认医学模型
        
    Returns:
        余弦相似度 (0-1之间的值，越大表示越相似)
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 如果未提供嵌入模型，使用默认模型
    if embedding_model is None:
        embedding_model = get_medical_embedding_model()
    
    # 获取两段文本的嵌入
    embedding1 = embedding_model.embed_query(text1)
    embedding2 = embedding_model.embed_query(text2)
    
    # 计算余弦相似度
    similarity = cosine_similarity(
        np.array(embedding1).reshape(1, -1),
        np.array(embedding2).reshape(1, -1)
    )[0][0]
    
    return float(similarity)


def batch_embed_texts(
    texts: List[str], 
    embedding_model: Optional[Embeddings] = None,
    chunk_size: int = 100
) -> List[List[float]]:
    """
    批量嵌入多个文本
    
    Args:
        texts: 要嵌入的文本列表
        embedding_model: 嵌入模型，如果未提供则使用默认医学模型
        chunk_size: 每批处理的文本数量
        
    Returns:
        嵌入向量列表
    """
    # 如果未提供嵌入模型，使用默认模型
    if embedding_model is None:
        embedding_model = get_medical_embedding_model()
    
    # 批量处理文本
    embeddings = []
    for i in range(0, len(texts), chunk_size):
        batch = texts[i:i+chunk_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
        print(f"已处理 {min(i+chunk_size, len(texts))}/{len(texts)} 个文本")
    
    return embeddings
