from typing import Dict, Any, Optional
from .base_tool import BaseTool, BaseToolMixin
import logging


class PizzaSearchTool(BaseTool, BaseToolMixin):
    """Tool for searching pizza information"""
    
    name = "pizza_search"
    description = """
    Search for pizza information including names, descriptions, prices, and general information.
    Use for questions about: what pizzas are available, pizza descriptions, prices, menu items.
    Input should be: pizza name or description of what customer is looking for.
    """
    
    def __init__(self, config: Dict[str, Any], vector_store=None):
        BaseToolMixin.__init__(self, config, vector_store)
        self.name = "pizza_search"
        self.description = """
        Search for pizza information including names, descriptions, prices, and general information.
        Use for questions about: what pizzas are available, pizza descriptions, prices, menu items.
        Input should be: pizza name or description of what customer is looking for.
        """
    
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
    
    def run(self, query: str) -> str:
        """Run the tool (alias for _run for compatibility)"""
        return self._run(query)
