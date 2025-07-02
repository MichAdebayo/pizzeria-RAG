from typing import Dict, Any, Optional
from .base_tool import BaseTool, BaseToolMixin
import logging


class NutritionInfoTool(BaseTool, BaseToolMixin):
    """Tool for retrieving nutritional information"""
    
    name = "nutrition_information"
    description = """
    Retrieve nutritional information and calorie data for pizzas.
    Use for questions about: calories, nutritional values, dietary information.
    Input should be: pizza name or nutrition question.
    """
    
    def __init__(self, config: Dict[str, Any], vector_store=None):
        BaseToolMixin.__init__(self, config, vector_store)
        self.name = "nutrition_information"
        self.description = """
        Retrieve nutritional information and calorie data for pizzas.
        Use for questions about: calories, nutritional values, dietary information.
        Input should be: pizza name or nutrition question.
        """
    
    def _run(self, query: str) -> str:
        """Retrieve nutrition information"""
        try:
            k = self.config.get('k', 3)
            docs = self._safe_search(query, k=k)
            
            if not docs:
                return "Je n'ai pas trouvé d'informations nutritionnelles pour cette recherche."
            
            response = "📊 Informations nutritionnelles:\n\n"
            formatted_info = self._format_documents(docs)
            response += formatted_info
            
            return response
            
        except Exception as e:
            self.logger.error(f"Nutrition lookup failed: {e}")
            return f"Désolé, une erreur s'est produite lors de la recherche nutritionnelle: {str(e)}"
    
    def run(self, query: str) -> str:
        """Run the tool (alias for _run for compatibility)"""
        return self._run(query)
