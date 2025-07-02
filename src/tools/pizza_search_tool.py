from typing import Dict, Any, Optional, Type
from .base_tool import PizzeriaBaseTool
import logging


class PizzaSearchTool(PizzeriaBaseTool):
    """Tool for searching pizza information"""
    
    name: str = "pizza_search"
    description: str = (
        "Search for pizza information including names, descriptions, prices, and general information. "
        "Use for questions about: what pizzas are available, pizza descriptions, prices, menu items. "
        "Input should be: pizza name or description of what customer is looking for."
    )
    
    def _run(self, query: str) -> str:
        """Search for pizza information"""
        try:
            k = self.config.get('k', 5)
            docs = self._safe_search(query, k=k)
            
            if not docs:
                return "Je n'ai pas trouvé d'information sur cette pizza. Pouvez-vous préciser votre demande?"
            
            response = "🍕 Voici les informations sur nos pizzas:\n\n"
            for i, doc in enumerate(docs, 1):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                response += f"{i}. {content}\n"
                
                # Add source if available
                if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                    response += f"   (Source: {doc.metadata['source']})\n"
                response += "\n"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Pizza search failed: {e}")
            return f"Désolé, une erreur s'est produite lors de la recherche: {str(e)}"
    
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
