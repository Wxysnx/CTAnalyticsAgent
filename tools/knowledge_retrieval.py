 """
知识检索工具: 使用LangChain RAG从医学知识库检索相关信息
"""
import os
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import VECTOR_DB_DIR, TOP_K_RETRIEVALS
from langchain_components.document_loaders import load_medical_knowledge_base
from langchain_components.embeddings import get_medical_embedding_model
from langchain_components.vectorstore import create_or_load_vectorstore
from langchain_components.retriever import (
    get_medical_knowledge_retriever,
    generate_multiple_queries,
    retrieve_medical_knowledge,
    build_medical_context
)


class MedicalKnowledgeRetrievalTool:
    """医学知识检索工具"""
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        retriever: Optional[BaseRetriever] = None,
        vector_db_dir: str = VECTOR_DB_DIR,
        collection_name: str = "medical_knowledge",
        rebuild_vectordb: bool = False
    ):
        """
        初始化医学知识检索工具
        
        Args:
            llm: 语言模型，用于查询生成和知识整合
            retriever: 检索器实例，如果提供则直接使用
            vector_db_dir: 向量数据库目录
            collection_name: 集合名称
            rebuild_vectordb: 是否重建向量数据库
        """
        self.llm = llm
        self.vector_db_dir = vector_db_dir
        self.collection_name = collection_name
        
        # 设置检索器
        if retriever is not None:
            self.retriever = retriever
        else:
            # 检查向量库是否存在
            vector_db_exists = os.path.exists(vector_db_dir) and len(os.listdir(vector_db_dir)) > 0
            
            if not vector_db_exists or rebuild_vectordb:
                print("向量数据库不存在或需要重建，正在创建...")
                # 加载医学知识
                documents = load_medical_knowledge_base()
                if not documents:
                    raise ValueError("无法加载医学知识文档")
                
                # 创建嵌入模型
                embedding_model = get_medical_embedding_model()
                
                # 创建向量存储和检索器
                self.retriever = get_medical_knowledge_retriever(
                    documents=documents,
                    embedding_model=embedding_model
                )
            else:
                print("加载现有向量数据库...")
                # 加载现有向量存储和检索器
                embedding_model = get_medical_embedding_model()
                vectorstore = create_or_load_vectorstore(
                    embedding_model=embedding_model,
                    persist_directory=vector_db_dir,
                    collection_name=collection_name
                )
                self.retriever = get_medical_knowledge_retriever(
                    vectorstore=vectorstore
                )
    
    def retrieve_knowledge(
        self,
        query: str,
        k: int = TOP_K_RETRIEVALS,
        return_documents: bool = False
    ) -> Union[str, List[Document]]:
        """
        检索相关医学知识
        
        Args:
            query: 查询文本
            k: 检索结果数量
            return_documents: 是否返回文档对象而非文本
            
        Returns:
            检索到的知识上下文或文档列表
        """
        # 检索相关文档
        documents = retrieve_medical_knowledge(query, self.retriever)
        
        # 限制结果数量
        documents = documents[:k]
        
        if return_documents:
            return documents
        else:
            # 构建上下文
            context = build_medical_context(documents)
            return context
    
    def retrieve_knowledge_from_ct_analysis(
        self,
        ct_analysis_result: Dict[str, Any],
        num_queries: int = 3,
        return_documents: bool = False
    ) -> Union[str, Dict[str, List[Document]]]:
        """
        基于CT分析结果检索相关医学知识
        
        Args:
            ct_analysis_result: CT分析结果
            num_queries: 生成的查询数量
            return_documents: 是否返回文档对象而非文本
            
        Returns:
            检索到的知识上下文或按查询分组的文档字典
        """
        # 提取CT分析描述
        if "summary" in ct_analysis_result:
            ct_description = ct_analysis_result["summary"].get("combined_description", "")
        elif "analysis" in ct_analysis_result:
            ct_description = ct_analysis_result["analysis"].get("combined_description", "")
        else:
            ct_description = str(ct_analysis_result)
        
        # 如果提供了LLM，生成多个查询
        if self.llm and ct_description:
            queries = generate_multiple_queries(ct_description, self.llm, num_queries)
        else:
            # 否则使用单个查询
            queries = [ct_description]
        
        all_documents = {}
        all_context_parts = []
        
        # 对每个查询执行检索
        for i, query in enumerate(queries):
            documents = retrieve_medical_knowledge(query, self.retriever)
            
            if return_documents:
                all_documents[f"query_{i}"] = documents
            else:
                context = build_medical_context(documents)
                all_context_parts.append(f"--- 查询 {i+1}: {query} ---\n\n{context}")
        
        if return_documents:
            return all_documents
        else:
            # 组合所有查询的结果
            combined_context = "\n\n".join(all_context_parts)
            return combined_context
    
    def integrate_knowledge_with_llm(
        self,
        ct_analysis_result: Dict[str, Any],
        knowledge_context: str
    ) -> str:
        """
        使用LLM整合CT分析结果和检索到的医学知识
        
        Args:
            ct_analysis_result: CT分析结果
            knowledge_context: 检索到的知识上下文
            
        Returns:
            整合后的知识总结
        """
        if not self.llm:
            raise ValueError("需要提供LLM模型才能整合知识")
        
        # 提取CT分析描述
        if "summary" in ct_analysis_result:
            ct_description = ct_analysis_result["summary"].get("combined_description", "")
        elif "analysis" in ct_analysis_result:
            ct_description = ct_analysis_result["analysis"].get("combined_description", "")
        else:
            ct_description = str(ct_analysis_result)
        
        # 创建提示模板
        template = """
        作为医学影像专家，请基于CT图像分析结果和检索到的医学知识，提供综合的医学解释。
        
        CT图像分析结果：
        {ct_description}
        
        相关医学知识：
        {knowledge_context}
        
        请提供综合的医学分析，包括可能的诊断、相关医学解释和临床意义。请使用专业但清晰的语言：
        """
        
        # 创建提示
        prompt = PromptTemplate(
            template=template,
            input_variables=["ct_description", "knowledge_context"]
        )
        
        # 创建链
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # 执行链
        result = chain.run(
            ct_description=ct_description,
            knowledge_context=knowledge_context
        )
        
        return result
