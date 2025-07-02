from typing import Dict, Any, Optional
import logging

try:
    from langchain.tools import BaseTool
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create a mock base tool for development
    class BaseTool:
        name = "base_tool"
        description = "Base tool"
        
        def run(self, query: str) -> str:
            return "Mock response"


class BaseToolMixin:
    """Base mixin for all pizzeria tools"""
    
    def __init__(self, config: Dict[str, Any], vector_store=None):
        self.config = config
        self.vector_store = vector_store
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set tool properties from config
        self.name = config.get('name', self.__class__.__name__.lower())
        self.description = config.get('description', 'Pizzeria tool')
    
    def _safe_search(self, query: str, k: int = 5) -> list:
        """Safely search vector store with fallbacks"""
        if not self.vector_store:
            self.logger.warning("No vector store available")
            return []
        
        try:
            retriever_type = self.config.get('retriever_type', 'similarity')
            
            if retriever_type == 'similarity':
                return self.vector_store.similarity_search(query, k=k)
            elif retriever_type == 'mmr':
                return self.vector_store.max_marginal_relevance_search(query, k=k)
            else:
                return self.vector_store.similarity_search(query, k=k)
                
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _format_documents(self, docs: list) -> str:
        """Format documents for response"""
        if not docs:
            return "Aucune information trouvée."
        
        formatted = []
        for doc in docs:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            # Add source if available
            if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                content += f"\n(Source: {doc.metadata['source']})"
            
            formatted.append(content)
        
        return "\n\n".join(formatted)
