from typing import Dict, List, Any, Optional
import logging


class AllergenParser:
    """Parser for allergen information documents"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def parse(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse allergen content"""
        text = content.get('text', '')
        tables = content.get('tables', [])
        
        parsed_data = {
            'allergen_info': self._extract_allergen_info(text, tables),
            'document_type': 'allergen_table'
        }
        
        return parsed_data
    
    def _extract_allergen_info(self, text: str, tables: List) -> Dict:
        """Extract allergen information from text and tables"""
        allergen_info = {
            'allergens_mentioned': [],
            'allergen_tables': []
        }
        
        # Extract mentioned allergens from text
        allergen_keywords = ['gluten', 'lactose', 'œuf', 'arachide', 'noix', 'soja', 'poisson']
        for keyword in allergen_keywords:
            if keyword.lower() in text.lower():
                allergen_info['allergens_mentioned'].append(keyword)
        
        # Process tables if available
        for table in tables:
            if self._is_allergen_table(table):
                allergen_info['allergen_tables'].append(table)
        
        return allergen_info
    
    def _is_allergen_table(self, table: Dict) -> bool:
        """Check if a table contains allergen information"""
        # Simple heuristic: look for allergen keywords in table data
        table_data = str(table.get('data', ''))
        allergen_keywords = ['gluten', 'lactose', 'allergen']
        return any(keyword in table_data.lower() for keyword in allergen_keywords)
