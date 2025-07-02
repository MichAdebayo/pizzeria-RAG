"""
Tools package for pizzeria RAG system
"""

# Import the base tool first
from .base_tool import PizzeriaBaseTool

# Import all tools for easy access
try:
    from .pizza_search_tool import PizzaSearchTool
    from .allergen_check_tool import AllergenCheckTool
    
    # Try to import the v2 tools first, fallback to original
    try:
        from .ingredient_lookup_tool_v2 import IngredientLookupTool
        from .nutrition_info_tool_v2 import NutritionInfoTool
    except ImportError:
        from .ingredient_lookup_tool import IngredientLookupTool
        from .nutrition_info_tool import NutritionInfoTool
    
    __all__ = [
        'PizzeriaBaseTool',
        'PizzaSearchTool',
        'AllergenCheckTool', 
        'IngredientLookupTool',
        'NutritionInfoTool'
    ]
except ImportError as e:
    # Handle import errors gracefully during development
    import logging
    logging.warning(f"Some tools could not be imported: {e}")
    __all__ = ['PizzeriaBaseTool']
