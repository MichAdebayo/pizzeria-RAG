from typing import Dict, List, Any, Optional
import logging
import re


class DocumentClassifier:
    """Classifies documents based on content patterns"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load classification patterns from config
        self.classification_patterns = self.config.get('document_classification', {})
    
    def classify(self, extracted_content: Dict[str, Any]) -> str:
        """Classify document type based on content"""
        text = extracted_content.get('text', '').lower()
        
        if not text:
            return 'unknown'
        
        # Score each document type
        scores = {}
        
        for doc_type, patterns in self.classification_patterns.items():
            scores[doc_type] = self._calculate_type_score(text, patterns)
        
        # Return the highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0.3:  # Minimum confidence threshold
                return best_type
        
        return 'unknown'
    
    def _calculate_type_score(self, text: str, patterns: Dict) -> float:
        """Calculate score for a specific document type"""
        score = 0.0
        
        # Check keywords
        keywords = patterns.get('keywords', [])
        for keyword in keywords:
            if keyword.lower() in text:
                score += 0.1
        
        # Check regex patterns
        regex_patterns = patterns.get('patterns', [])
        for pattern in regex_patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 0.2
            except re.error:
                self.logger.warning(f"Invalid regex pattern: {pattern}")
        
        return min(1.0, score)


class PizzaParser:
    """Parser for pizza menu documents"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def parse(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse pizza menu content"""
        text = content.get('text', '')
        
        parsed_data = {
            'pizzas': self._extract_pizzas(text),
            'prices': self._extract_prices(text),
            'document_type': 'menu_catalog'
        }
        
        return parsed_data
    
    def _extract_pizzas(self, text: str) -> List[Dict]:
        """Extract pizza information from text"""
        pizzas = []
        
        # Simple pizza name extraction
        pizza_patterns = [
            r'pizza\s+([a-zA-ZÀ-ÿ\s]+?)(?=\s*[€\d]|$)',
            r'^([A-ZÀ-Ÿ][a-zA-ZÀ-ÿ\s]+?)(?=\s*[-–])'
        ]
        
        for pattern in pizza_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                pizza_name = match.strip()
                if len(pizza_name) > 2:  # Filter out very short matches
                    pizzas.append({
                        'name': pizza_name,
                        'type': 'pizza'
                    })
        
        return pizzas
    
    def _extract_prices(self, text: str) -> List[Dict]:
        """Extract price information from text"""
        prices = []
        
        price_patterns = [
            r'([€]?\s*\d+[.,]\d{2}\s*€?)',
            r'(\d+[.,]\d{2}\s*euros?)'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                prices.append({
                    'price': match.strip(),
                    'type': 'price'
                })
        
        return prices


class AllergenParser:
    """Parser for allergen information documents"""
    
    def __init__(self, config: Dict = None):
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


class RecipeParser:
    """Parser for recipe documents"""
    
    def __init__(self, config: Dict = None):
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
