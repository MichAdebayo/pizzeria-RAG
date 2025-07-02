from typing import Dict, Any
from .base_tool import PizzeriaBaseTool
import logging


class IngredientLookupTool(PizzeriaBaseTool):
    """Tool for looking up ingredient information"""
    
    name: str = "ingredient_lookup"
    description: str = (
        "Look up detailed ingredient information for pizzas and recipes. "
        "Use for questions about: what ingredients are in a pizza, recipe details, preparation methods. "
        "Input should be: pizza name or ingredient name you want to know about."
    )
    
    def _run(self, query: str) -> str:
        """Look up ingredient information"""
        try:
            k = self.config.get('k', 5)
            docs = self._safe_search(query, k=k)
            
            if not docs:
                return "Je n'ai pas trouvé d'information sur ces ingrédients. Pouvez-vous préciser votre demande?"
            
            response = "🥕 Voici les informations sur les ingrédients:\n\n"
            for i, doc in enumerate(docs, 1):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                response += f"{i}. {content}\n"
                
                # Add source if available
                if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                    response += f"   (Source: {doc.metadata['source']})\n"
                response += "\n"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ingredient lookup failed: {e}")
            return f"Désolé, une erreur s'est produite lors de la recherche: {str(e)}"
