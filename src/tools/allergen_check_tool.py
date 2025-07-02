from typing import Dict, Any, Optional
from .base_tool import BaseTool, BaseToolMixin
import logging


class AllergenCheckTool(BaseTool, BaseToolMixin):
    """CRITICAL SAFETY TOOL: Check allergen information for pizzas"""
    
    name = "allergen_safety_check"
    description = """
    CRITICAL SAFETY TOOL: Check allergen information for pizzas.
    Use this tool for ANY question about allergies, allergens, or dietary restrictions.
    Input should be: pizza name and allergen concern (e.g., "Margherita gluten")
    """
    
    def __init__(self, config: Dict[str, Any], vector_store=None):
        BaseToolMixin.__init__(self, config, vector_store)
        self.name = "allergen_safety_check"
        self.safety_threshold = config.get('confidence_threshold', 0.99)
        self.safety_mode = config.get('safety_mode', True)
        self.description = """
        CRITICAL SAFETY TOOL: Check allergen information for pizzas.
        Use this tool for ANY question about allergies, allergens, or dietary restrictions.
        Input should be: pizza name and allergen concern (e.g., "Margherita gluten")
        """
    
    def _run(self, query: str) -> str:
        """Execute allergen safety check"""
        try:
            # Enhanced search for allergen information
            k = self.config.get('k', 3)
            docs = self._safe_search(query, k=k)
            
            if not docs:
                return self._safety_fallback_response(query)
            
            # For now, we'll return the information with safety warnings
            # In a real implementation, you'd check confidence scores
            allergen_info = self._format_documents(docs)
            return self._format_safety_response(allergen_info, query)
            
        except Exception as e:
            self.logger.error(f"Allergen check failed: {e}")
            return self._safety_fallback_response(query)
    
    def _safety_fallback_response(self, query: str) -> str:
        """Return safety fallback when information is uncertain"""
        return (
            "⚠️ IMPORTANT: Je ne peux pas confirmer ces informations allergéniques "
            "avec une certitude suffisante. Pour votre sécurité, veuillez contacter "
            "directement le restaurant au [numéro] pour confirmer les allergènes. "
            "La sécurité alimentaire est notre priorité absolue."
        )
    
    def _format_safety_response(self, allergen_info: str, query: str) -> str:
        """Format response with safety disclaimers"""
        response = f"✅ Information allergénique trouvée:\n\n{allergen_info}\n\n"
        response += "⚠️ IMPORTANT: Ces informations proviennent de notre base documentaire. "
        response += "En cas d'allergie sévère ou de doute, contactez directement le restaurant "
        response += "pour confirmation. Votre sécurité est notre priorité."
        return response
    
    def run(self, query: str) -> str:
        """Run the tool (alias for _run for compatibility)"""
        return self._run(query)
