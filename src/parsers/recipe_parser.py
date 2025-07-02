from typing import Dict, List, Any, Optional
import logging
import re


class RecipeParser:
    """Parser for recipe documents"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def parse(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse recipe content"""
        text = content.get('text', '')
        
        parsed_data = {
            'recipe_info': self._extract_recipe_info(text),
            'document_type': 'recipe'
        }
        
        return parsed_data
    
    def _extract_recipe_info(self, text: str) -> Dict:
        """Extract recipe information from text"""
        recipe_info = {
            'ingredients': self._extract_ingredients(text),
            'instructions': self._extract_instructions(text),
            'title': self._extract_title(text)
        }
        
        return recipe_info
    
    def _extract_ingredients(self, text: str) -> List[str]:
        """Extract ingredients from text"""
        ingredients = []
        
        # Look for ingredient sections
        ingredient_section = re.search(
            r'ingrédients?\s*:?(.*?)(?=préparation|instructions|étapes|$)',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if ingredient_section:
            ingredient_text = ingredient_section.group(1)
            # Split by lines and clean up
            lines = ingredient_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 3:
                    ingredients.append(line)
        
        return ingredients
    
    def _extract_instructions(self, text: str) -> List[str]:
        """Extract cooking instructions from text"""
        instructions = []
        
        # Look for instruction sections
        instruction_section = re.search(
            r'(?:préparation|instructions|étapes)\s*:?(.*?)$',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if instruction_section:
            instruction_text = instruction_section.group(1)
            # Split by numbered steps or lines
            lines = instruction_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    instructions.append(line)
        
        return instructions
    
    def _extract_title(self, text: str) -> str:
        """Extract recipe title from text"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first few lines
            line = line.strip()
            if line and len(line) < 100:  # Reasonable title length
                return line
        return "Recette"
