 """
文档加载器模块: 配置和提供医学文档的加载功能
"""
import os
from typing import List, Dict, Any, Optional, Union
import glob

from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader,
    DirectoryLoader,
    BSHTMLLoader
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import MEDICAL_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf_document(file_path: str) -> List[Document]:
    """
    加载PDF文档
    
    Args:
        file_path: PDF文件路径
        
    Returns:
        文档对象列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text_document(file_path: str) -> List[Document]:
    """
    加载文本文档
    
    Args:
        file_path: 文本文件路径
        
    Returns:
        文档对象列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def load_markdown_document(file_path: str) -> List[Document]:
    """
    加载Markdown文档
    
    Args:
        file_path: Markdown文件路径
        
    Returns:
        文档对象列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()


def load_html_document(file_path: str) -> List[Document]:
    """
    加载HTML文档
    
    Args:
        file_path: HTML文件路径
        
    Returns:
        文档对象列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    loader = BSHTMLLoader(file_path)
    return loader.load()


def load_directory(
    directory_path: str, 
    glob_pattern: str = "**/*.*", 
    show_progress: bool = True
) -> List[Document]:
    """
    加载目录中的所有文档
    
    Args:
        directory_path: 目录路径
        glob_pattern: 文件匹配模式
        show_progress: 是否显示进度条
        
    Returns:
        文档对象列表
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录不存在: {directory_path}")
    
    # 创建不同类型文档的加载器
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".html": BSHTMLLoader,
        ".htm": BSHTMLLoader,
    }
    
    # 使用DirectoryLoader加载目录中的所有文档
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        loader_cls=lambda file_path: select_loader(file_path, loaders),
        show_progress=show_progress
    )
    
    return loader.load()


def select_loader(file_path: str, loaders: Dict[str, BaseLoader]) -> BaseLoader:
    """
    根据文件扩展名选择合适的加载器
    
    Args:
        file_path: 文件路径
        loaders: 加载器字典
        
    Returns:
        合适的文档加载器
    """
    # 获取文件扩展名
    ext = os.path.splitext(file_path)[1].lower()
    
    # 选择合适的加载器
    if ext in loaders:
        return loaders[ext](file_path)
    else:
        # 默认使用TextLoader
        return TextLoader(file_path, encoding='utf-8')


def split_documents(
    documents: List[Document], 
    chunk_size: int = CHUNK_SIZE, 
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    分割文档为较小的块
    
    Args:
        documents: 要分割的文档列表
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        
    Returns:
        分割后的文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)


def load_medical_knowledge_base(
    directory: str = MEDICAL_DOCS_DIR, 
    glob_pattern: str = "**/*.*",
    use_splitter: bool = True,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    加载医学知识库中的所有文档
    
    Args:
        directory: 医学文档目录
        glob_pattern: 文件匹配模式
        use_splitter: 是否使用文本分割器
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        
    Returns:
        处理后的文档列表
    """
    # 加载目录中的所有文档
    try:
        documents = load_directory(directory, glob_pattern)
        print(f"成功加载 {len(documents)} 个文档")
        
        # 如果需要分割文档
        if use_splitter:
            documents = split_documents(documents, chunk_size, chunk_overlap)
            print(f"分割后得到 {len(documents)} 个文档块")
        
        return documents
    
    except Exception as e:
        print(f"加载医学知识库时出错: {e}")
        return []


def load_medical_documents_by_type(
    doc_type: str, 
    directory: str = MEDICAL_DOCS_DIR, 
    use_splitter: bool = True
) -> List[Document]:
    """
    按类型加载医学文档
    
    Args:
        doc_type: 文档类型 ("radiology", "pathology", "general", etc.)
        directory: 基础目录
        use_splitter: 是否使用文本分割器
        
    Returns:
        加载的文档列表
    """
    # 确定类型目录
    type_dir = os.path.join(directory, doc_type)
    
    # 如果类型目录不存在，尝试搜索包含类型名称的文件
    if not os.path.exists(type_dir):
        # 构建文件匹配模式
        pattern = f"**/*{doc_type}*.*"
        return load_medical_knowledge_base(directory, pattern, use_splitter)
    
    # 如果类型目录存在，加载该目录下的所有文档
    return load_medical_knowledge_base(type_dir, "**/*.*", use_splitter)
