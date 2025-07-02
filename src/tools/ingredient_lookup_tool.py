from typing import Dict, Any, Optional
from .base_tool import BaseTool, BaseToolMixin
import logging


class IngredientLookupTool(BaseTool, BaseToolMixin):
    """Tool for looking up detailed ingredient information"""
    
    name = "ingredient_lookup"
    description = """
    Get detailed ingredient lists and preparation information for pizzas.
    Use for questions about: ingredients, recipe details, preparation methods.
    Input should be: pizza name or ingredient question.
    """
    
    def __init__(self, config: Dict[str, Any], vector_store=None):
        BaseToolMixin.__init__(self, config, vector_store)
        self.name = "ingredient_lookup"
        self.description = """
        Get detailed ingredient lists and preparation information for pizzas.
        Use for questions about: ingredients, recipe details, preparation methods.
        Input should be: pizza name or ingredient question.
        """
    
    def _run(self, query: str) -> str:
        """Look up ingredient information"""
        try:
            k = self.config.get('k', 4)
            docs = self._safe_search(query, k=k)
            
            if not docs:
                return "Je n'ai pas trouvé d'informations sur les ingrédients pour cette recherche."
            
            response = "📋 Informations sur les ingrédients:\n\n"
            formatted_info = self._format_documents(docs)
            response += formatted_info
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ingredient lookup failed: {e}")
            return f"Désolé, une erreur s'est produite lors de la recherche d'ingrédients: {str(e)}"
    
    def run(self, query: str) -> str:
        """Run the tool (alias for _run for compatibility)"""
        return self._run(query)
