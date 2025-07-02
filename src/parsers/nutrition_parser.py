from typing import Dict, List, Any, Optional
import logging
import re


class NutritionParser:
    """Parser for nutrition information documents"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def parse(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse nutrition content"""
        text = content.get('text', '')
        tables = content.get('tables', [])
        
        parsed_data = {
            'nutrition_info': self._extract_nutrition_info(text, tables),
            'document_type': 'nutrition_table'
        }
        
        return parsed_data
    
    def _extract_nutrition_info(self, text: str, tables: List) -> Dict:
        """Extract nutrition information from text and tables"""
        nutrition_info = {
            'calories': self._extract_calories(text),
            'nutrients': self._extract_nutrients(text),
            'nutrition_tables': []
        }
        
        # Process tables if available
        for table in tables:
            if self._is_nutrition_table(table):
                nutrition_info['nutrition_tables'].append(table)
        
        return nutrition_info
    
    def _extract_calories(self, text: str) -> List[Dict]:
        """Extract calorie information from text"""
        calories = []
        
        calorie_patterns = [
            r'(\d+)\s*(?:kcal|calories?)',
            r'(?:kcal|calories?)\s*[:\s]*(\d+)',
            r'(\d+)\s*cal\b'
        ]
        
        for pattern in calorie_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                calories.append({
                    'value': match,
                    'unit': 'kcal'
                })
        
        return calories
    
    def _extract_nutrients(self, text: str) -> List[Dict]:
        """Extract nutrient information from text"""
        nutrients = []
        
        nutrient_patterns = [
            r'protéines?\s*[:\s]*(\d+(?:[.,]\d+)?)\s*g',
            r'glucides?\s*[:\s]*(\d+(?:[.,]\d+)?)\s*g',
            r'lipides?\s*[:\s]*(\d+(?:[.,]\d+)?)\s*g',
            r'fibres?\s*[:\s]*(\d+(?:[.,]\d+)?)\s*g'
        ]
        
        nutrient_names = ['protéines', 'glucides', 'lipides', 'fibres']
        
        for i, pattern in enumerate(nutrient_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                nutrients.append({
                    'name': nutrient_names[i],
                    'value': match,
                    'unit': 'g'
                })
        
        return nutrients
    
    def _is_nutrition_table(self, table: Dict) -> bool:
        """Check if a table contains nutrition information"""
        table_data = str(table.get('data', ''))
        nutrition_keywords = ['calories', 'kcal', 'protéines', 'glucides', 'lipides', 'nutrition']
        return any(keyword in table_data.lower() for keyword in nutrition_keywords)
