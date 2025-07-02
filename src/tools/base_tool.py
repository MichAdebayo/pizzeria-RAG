from typing import Dict, Any, Optional
import logging

# Clean separation of LangChain and mock implementations
try:
    from langchain.tools import BaseTool as LangChainBaseTool
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create a mock base class that matches LangChain's interface
    class LangChainBaseTool:
        """Mock LangChain BaseTool for development"""
        name: str = "mock_tool"
        description: str = "Mock tool"
        
        def run(self, query: str) -> str:
            return "Mock response"


# Define the base tool class with proper inheritance
if LANGCHAIN_AVAILABLE:
    class PizzeriaBaseTool(LangChainBaseTool):
        """Base tool class for all pizzeria tools with LangChain integration"""
        
        name: str = "base_pizzeria_tool"
        description: str = "Base pizzeria tool"
        
        def __init__(self, config: Dict[str, Any] = None, vector_store=None, **kwargs):
            # Don't pass config and vector_store to super() as they aren't Pydantic fields
            super().__init__(**kwargs)
            
            # Store as instance variables (not Pydantic fields)
            self._config = config or {}
            self._vector_store = vector_store
            self._logger = logging.getLogger(self.__class__.__name__)
            
            # Update name and description from config if provided
            if self._config and 'name' in self._config:
                self.name = self._config['name']
            if self._config and 'description' in self._config:
                self.description = self._config['description']
        
        @property
        def config(self) -> Dict[str, Any]:
            return self._config
        
        @property
        def vector_store(self):
            return self._vector_store
        
        @property
        def logger(self):
            return self._logger
        
        def _safe_search(self, query: str, k: int = 5) -> list:
            """Safely search vector store with fallbacks"""
            if not self.vector_store:
                self.logger.warning("No vector store available")
                return []
            
            try:
                retriever_type = self.config.get('retriever_type', 'similarity')
                
                if retriever_type == 'mmr':
                    return self.vector_store.max_marginal_relevance_search(query, k=k)
                else:  # Default to similarity search
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
        
        def _run(self, query: str) -> str:
            """Override this method in subclasses"""
            return f"Base tool response for: {query}"

else:
    # Mock implementation for development without LangChain
    class PizzeriaBaseTool(LangChainBaseTool):
        """Mock base tool class for development without LangChain"""
        
        name: str = "mock_pizzeria_tool"
        description: str = "Mock pizzeria tool for development"
        
        def __init__(self, config: Dict[str, Any] = None, vector_store=None, **kwargs):
            self._config = config or {}
            self._vector_store = vector_store
            self._logger = logging.getLogger(self.__class__.__name__)
            
            # Update name and description from config if provided
            if self._config and 'name' in self._config:
                self.name = self._config['name']
            if self._config and 'description' in self._config:
                self.description = self._config['description']
        
        @property
        def config(self) -> Dict[str, Any]:
            return self._config
        
        @property
        def vector_store(self):
            return self._vector_store
        
        @property
        def logger(self):
            return self._logger
        
        def _safe_search(self, query: str, k: int = 5) -> list:
            """Mock search method"""
            self.logger.info(f"Mock search for: {query}")
            return []
        
        def _format_documents(self, docs: list) -> str:
            """Mock format method"""
            return "Mock formatted documents"
        
        def _run(self, query: str) -> str:
            """Mock run method"""
            return f"Mock response for: {query}"
        
        def run(self, query: str) -> str:
            return self._run(query)
