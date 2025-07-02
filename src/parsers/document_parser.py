from typing import Dict, List, Any, Optional
import logging
import re


class DocumentParser:
    """Classifies documents based on content patterns"""
    
    def __init__(self, config: Optional[Dict] = None):
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
