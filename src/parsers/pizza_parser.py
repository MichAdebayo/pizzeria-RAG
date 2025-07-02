from typing import Dict, List, Any, Optional
import logging
import re


class PizzaParser:
    """Parser for pizza menu documents"""
    
    def __init__(self, config: Optional[Dict] = None):
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
