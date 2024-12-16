 """
检索器模块: 配置和提供高级文档检索功能
"""
from typing import List, Dict, Any, Optional, Callable, Union

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import TOP_K_RETRIEVALS, SIMILARITY_THRESHOLD
from langchain_components.embeddings import get_medical_embedding_model
from langchain_components.vectorstore import create_or_load_vectorstore


def get_basic_retriever(
    vectorstore: Chroma,
    search_kwargs: Optional[Dict[str, Any]] = None
) -> BaseRetriever:
    """
    获取基本向量检索器
    
    Args:
        vectorstore: 向量存储实例
        search_kwargs: 搜索参数
        
    Returns:
        基本检索器
    """
    if search_kwargs is None:
        search_kwargs = {"k": TOP_K_RETRIEVALS}
    
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def rerank_documents(
    documents: List[Document],
    query: str,
    embedding_model: Optional[Embeddings] = None,
    top_k: int = TOP_K_RETRIEVALS,
    threshold: float = SIMILARITY_THRESHOLD
) -> List[Document]:
    """
    对检索到的文档进行重排序
    
    Args:
        documents: 检索到的文档列表
        query: 查询文本
        embedding_model: 嵌入模型
        top_k: 保留的结果数量
        threshold: 相似度阈值，低于此值的结果将被过滤
        
    Returns:
        重排序后的文档列表
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 如果没有文档，返回空列表
    if not documents:
        return []
    
    # 如果未提供嵌入模型，使用默认医学模型
    if embedding_model is None:
        embedding_model = get_medical_embedding_model()
    
    # 嵌入查询
    query_embedding = embedding_model.embed_query(query)
    
    # 嵌入每个文档的内容
    doc_embeddings = []
    for doc in documents:
        doc_embedding = embedding_model.embed_query(doc.page_content)
        doc_embeddings.append(doc_embedding)
    
    # 计算余弦相似度
    similarities = []
    for doc_embedding in doc_embeddings:
        similarity = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(doc_embedding).reshape(1, -1)
        )[0][0]
        similarities.append(similarity)
    
    # 创建文档-相似度对
    doc_similarity_pairs = list(zip(documents, similarities))
    
    # 按相似度排序
    sorted_pairs = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)
    
    # 过滤掉相似度低于阈值的文档
    filtered_pairs = [(doc, sim) for doc, sim in sorted_pairs if sim >= threshold]
    
    # 限制结果数量
    filtered_pairs = filtered_pairs[:top_k]
    
    # 提取排序后的文档
    reranked_documents = [doc for doc, _ in filtered_pairs]
    
    return reranked_documents


def get_contextual_compression_retriever(
    vectorstore: Chroma,
    llm: BaseLanguageModel,
    search_kwargs: Optional[Dict[str, Any]] = None
) -> ContextualCompressionRetriever:
    """
    获取上下文压缩检索器，可以提取相关信息片段
    
    Args:
        vectorstore: 向量存储实例
        llm: 语言模型
        search_kwargs: 搜索参数
        
    Returns:
        上下文压缩检索器
    """
    # 创建基本检索器
    base_retriever = get_basic_retriever(vectorstore, search_kwargs)
    
    # 创建LLM提取器
    compressor = LLMChainExtractor.from_llm(llm)
    
    # 创建上下文压缩检索器
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return retriever


def get_medical_knowledge_retriever(
    documents: Optional[List[Document]] = None,
    embedding_model: Optional[Embeddings] = None,
    vectorstore: Optional[Chroma] = None,
    k: int = TOP_K_RETRIEVALS
) -> BaseRetriever:
    """
    获取医学知识检索器
    
    Args:
        documents: 文档列表(如果需要创建新的向量存储)
        embedding_model: 嵌入模型
        vectorstore: 现有向量存储(如果已有)
        k: 检索结果数量
        
    Returns:
        医学知识检索器
    """
    # 获取向量存储
    if vectorstore is None:
        vectorstore = create_or_load_vectorstore(
            documents=documents,
            embedding_model=embedding_model
        )
    
    # 创建检索器
    return get_basic_retriever(
        vectorstore=vectorstore,
        search_kwargs={"k": k}
    )


def create_medical_query_rewriter(
    llm: BaseLanguageModel
) -> Callable[[str], str]:
    """
    创建医学查询重写函数，用于优化检索查询
    
    Args:
        llm: 语言模型
        
    Returns:
        查询重写函数
    """
    template = """
    你是一位医学检索专家，请将以下查询重写为更有效的医学检索查询。
    保留所有关键的医学术语，添加可能相关的同义词，移除不必要的词语。
    
    原始查询: {query}
    
    重写后的查询:
    """
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    def rewrite_query(query: str) -> str:
        """
        重写查询以优化检索结果
        
        Args:
            query: 原始查询
            
        Returns:
            重写后的查询
        """
        try:
            rewritten_query = chain.run(query=query).strip()
            return rewritten_query
        except Exception as e:
            print(f"重写查询时出错: {e}")
            return query
    
    return rewrite_query


def generate_multiple_queries(
    ct_findings: str,
    llm: BaseLanguageModel,
    num_queries: int = 3
) -> List[str]:
    """
    基于CT发现生成多个检索查询
    
    Args:
        ct_findings: CT发现描述
        llm: 语言模型
        num_queries: 生成的查询数量
        
    Returns:
        查询列表
    """
    template = """
    基于以下CT图像分析结果，生成{num_queries}个不同的医学检索查询，每个查询关注不同的关键医学发现或可能的诊断。
    查询应当简洁、精确，包含关键的医学术语。
    
    CT分析结果:
    {ct_findings}
    
    查询列表(每行一个查询):
    """
    
    prompt = PromptTemplate(
        input_variables=["ct_findings", "num_queries"],
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run(ct_findings=ct_findings, num_queries=num_queries).strip()
        queries = [q.strip() for q in result.split('\n') if q.strip()]
        return queries[:num_queries]  # 确保不超过请求的查询数量
    except Exception as e:
        print(f"生成查询时出错: {e}")
        # 返回一个基本查询
        return [f"CT影像医学分析: {ct_findings[:100]}..."]


def retrieve_medical_knowledge(
    query: str,
    retriever: BaseRetriever,
    rerank: bool = True,
    embedding_model: Optional[Embeddings] = None
) -> List[Document]:
    """
    检索医学知识
    
    Args:
        query: 查询文本
        retriever: 检索器
        rerank: 是否对结果进行重排序
        embedding_model: 用于重排序的嵌入模型
        
    Returns:
        检索到的文档列表
    """
    # 执行检索
    documents = retriever.get_relevant_documents(query)
    
    # 如果需要重排序
    if rerank and documents:
        documents = rerank_documents(
            documents=documents,
            query=query,
            embedding_model=embedding_model
        )
    
    return documents


def build_medical_context(
    documents: List[Document],
    max_len: int = 4000
) -> str:
    """
    从检索文档构建医学上下文
    
    Args:
        documents: 检索到的文档列表
        max_len: 上下文最大长度
        
    Returns:
        构建的上下文文本
    """
    if not documents:
        return "无相关医学知识。"
    
    context_parts = []
    current_len = 0
    
    for i, doc in enumerate(documents):
        content = doc.page_content.strip()
        source = doc.metadata.get('source', f'来源 {i+1}')
        
        # 格式化为带有来源的段落
        formatted_content = f"[{source}]: {content}"
        
        # 检查是否会超过最大长度
        if current_len + len(formatted_content) > max_len:
            # 如果还没有添加任何内容，添加第一个文档的截断版本
            if not context_parts:
                truncated = formatted_content[:max_len]
                context_parts.append(truncated)
            break
        
        context_parts.append(formatted_content)
        current_len += len(formatted_content)
    
    return "\n\n".join(context_parts)
