from typing import Dict, Any, Optional
import logging


class QualityAssessment:
    """Assesses the quality of extracted content"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def assess(self, parsed_content: Dict[str, Any]) -> float:
        """Assess overall quality of parsed content"""
        
        text = parsed_content.get('text', '')
        document_type = parsed_content.get('document_type', 'unknown')
        
        if not text:
            return 0.0
        
        # Calculate different quality metrics
        metrics = {
            'text_length': self._assess_text_length(text),
            'text_coherence': self._assess_text_coherence(text),
            'domain_relevance': self._assess_domain_relevance(text),
            'structure_quality': self._assess_structure_quality(parsed_content)
        }
        
        # Weight the metrics
        weights = {
            'text_length': 0.2,
            'text_coherence': 0.3,
            'domain_relevance': 0.4,
            'structure_quality': 0.1
        }
        
        # Calculate weighted average
        total_score = sum(metrics[key] * weights[key] for key in metrics)
        
        return min(1.0, max(0.0, total_score))
    
    def _assess_text_length(self, text: str) -> float:
        """Assess if text has reasonable length"""
        word_count = len(text.split())
        
        if word_count < 10:
            return 0.1
        elif word_count < 50:
            return 0.5
        elif word_count < 200:
            return 0.8
        else:
            return 1.0
    
    def _assess_text_coherence(self, text: str) -> float:
        """Assess text coherence (basic heuristics)"""
        score = 0.5  # Base score
        
        # Check for reasonable sentence structure
        sentences = text.split('.')
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_sentence_length <= 30:  # Reasonable sentence length
                score += 0.2
        
        # Check for reasonable character distribution
        if len(set(text.lower())) > 10:  # At least 10 different characters
            score += 0.2
        
        # Check for common OCR errors or garbled text
        garbled_indicators = ['@@', '##', '||', '???']
        if not any(indicator in text for indicator in garbled_indicators):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_domain_relevance(self, text: str) -> float:
        """Assess relevance to pizzeria domain"""
        text_lower = text.lower()
        
        # Pizza-related terms
        pizza_terms = [
            'pizza', 'pizzeria', 'margherita', 'quattro', 'fromage', 'tomate',
            'ingrédient', 'ingredient', 'allergen', 'allergène', 'gluten', 'lactose',
            'recette', 'recipe', 'préparation', 'cuisson', 'menu', 'carte', 'prix', 'price'
        ]
        
        found_terms = sum(1 for term in pizza_terms if term in text_lower)
        relevance_score = min(1.0, found_terms / 5)  # Normalize to 1.0
        
        return relevance_score
    
    def _assess_structure_quality(self, parsed_content: Dict[str, Any]) -> float:
        """Assess quality of parsed structure"""
        score = 0.5  # Base score
        
        # Check if document type was identified
        if parsed_content.get('document_type', 'unknown') != 'unknown':
            score += 0.3
        
        # Check for structured content
        if 'pizzas' in parsed_content or 'allergen_info' in parsed_content or 'recipe_info' in parsed_content:
            score += 0.2
        
        return min(1.0, score)
