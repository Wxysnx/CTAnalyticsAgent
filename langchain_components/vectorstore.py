 """
向量存储模块: 提供基于Chroma的向量存储实现
"""
import os
from typing import List, Dict, Any, Optional, Union

from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import VECTOR_DB_DIR
from langchain_components.embeddings import get_medical_embedding_model


def create_vectorstore(
    documents: List[Document],
    embedding_model: Optional[Embeddings] = None,
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "medical_knowledge"
) -> Chroma:
    """
    创建向量存储
    
    Args:
        documents: 要存储的文档列表
        embedding_model: 嵌入模型，如果未提供则使用默认医学模型
        persist_directory: 向量存储持久化目录
        collection_name: 集合名称
        
    Returns:
        Chroma向量存储实例
    """
    # 如果未提供嵌入模型，使用默认模型
    if embedding_model is None:
        embedding_model = get_medical_embedding_model()
    
    # 创建向量存储
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    # 持久化到磁盘
    vectorstore.persist()
    
    return vectorstore


def load_vectorstore(
    persist_directory: str = VECTOR_DB_DIR,
    embedding_model: Optional[Embeddings] = None,
    collection_name: str = "medical_knowledge"
) -> Chroma:
    """
    加载现有向量存储
    
    Args:
        persist_directory: 向量存储持久化目录
        embedding_model: 嵌入模型，如果未提供则使用默认医学模型
        collection_name: 集合名称
        
    Returns:
        Chroma向量存储实例
    """
    # 如果未提供嵌入模型，使用默认模型
    if embedding_model is None:
        embedding_model = get_medical_embedding_model()
    
    # 检查向量存储目录是否存在
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"向量存储目录不存在: {persist_directory}")
    
    # 加载向量存储
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )


def create_or_load_vectorstore(
    documents: Optional[List[Document]] = None,
    embedding_model: Optional[Embeddings] = None,
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "medical_knowledge",
    recreate: bool = False
) -> Chroma:
    """
    创建或加载向量存储
    
    Args:
        documents: 要存储的文档列表(如果需要创建新的向量存储)
        embedding_model: 嵌入模型，如果未提供则使用默认医学模型
        persist_directory: 向量存储持久化目录
        collection_name: 集合名称
        recreate: 是否强制重新创建向量存储
        
    Returns:
        Chroma向量存储实例
    """
    # 如果未提供嵌入模型，使用默认模型
    if embedding_model is None:
        embedding_model = get_medical_embedding_model()
    
    # 检查向量存储是否已存在
    vector_db_exists = os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0
    
    if recreate or not vector_db_exists:
        # 如果需要重新创建或不存在，创建新的向量存储
        if documents is None:
            raise ValueError("创建新的向量存储需要提供文档")
        
        # 如果目录已存在但要重新创建，清空目录
        if vector_db_exists and recreate:
            import shutil
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
        
        return create_vectorstore(
            documents=documents,
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    else:
        # 加载现有向量存储
        return load_vectorstore(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            collection_name=collection_name
        )


def add_documents_to_vectorstore(
    vectorstore: Chroma,
    documents: List[Document]
) -> None:
    """
    向现有向量存储添加文档
    
    Args:
        vectorstore: 向量存储实例
        documents: 要添加的文档列表
    """
    vectorstore.add_documents(documents)
    vectorstore.persist()


def search_similar_documents(
    vectorstore: Union[Chroma, VectorStore],
    query: str,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    搜索相似文档
    
    Args:
        vectorstore: 向量存储实例
        query: 查询文本
        k: 返回的结果数量
        filter: 过滤条件
        
    Returns:
        相似文档列表
    """
    return vectorstore.similarity_search(query, k=k, filter=filter)


def search_similar_with_scores(
    vectorstore: Union[Chroma, VectorStore],
    query: str,
    k: int = 5
) -> List[tuple[Document, float]]:
    """
    搜索相似文档并返回相似度分数
    
    Args:
        vectorstore: 向量存储实例
        query: 查询文本
        k: 返回的结果数量
        
    Returns:
        包含文档和相似度分数的元组列表
    """
    return vectorstore.similarity_search_with_score(query, k=k)
