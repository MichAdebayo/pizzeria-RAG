from typing import Dict, Any
from .base_tool import PizzeriaBaseTool
import logging


class NutritionInfoTool(PizzeriaBaseTool):
    """Tool for providing nutrition information"""
    
    name: str = "nutrition_info"
    description: str = (
        "Provide nutrition information for pizzas including calories, macronutrients, and dietary information. "
        "Use for questions about: calories, carbs, protein, fat, dietary values, nutritional content. "
        "Input should be: pizza name or specific nutrition question."
    )
    
    def _run(self, query: str) -> str:
        """Provide nutrition information"""
        try:
            k = self.config.get('k', 5)
            docs = self._safe_search(query, k=k)
            
            if not docs:
                return "Je n'ai pas trouvé d'information nutritionnelle. Pouvez-vous préciser votre demande?"
            
            response = "📊 Voici les informations nutritionnelles:\n\n"
            for i, doc in enumerate(docs, 1):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                response += f"{i}. {content}\n"
                
                # Add source if available
                if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                    response += f"   (Source: {doc.metadata['source']})\n"
                response += "\n"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Nutrition info lookup failed: {e}")
            return f"Désolé, une erreur s'est produite lors de la recherche: {str(e)}"
