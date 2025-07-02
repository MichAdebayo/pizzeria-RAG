from typing import Dict, List, Any, Optional
import logging
import re


class DocumentChunker:
    """Creates chunks from processed documents for vector storage"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default chunking parameters
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
    
    def create_chunks(self, parsed_content: Dict[str, Any], doc_type: str) -> List[Dict]:
        """Create chunks based on document type and content"""
        
        text = parsed_content.get('text', '')
        if not text:
            return []
        
        # Choose chunking strategy based on document type
        if doc_type == 'allergen_table':
            return self._chunk_allergen_content(parsed_content)
        elif doc_type == 'recipe':
            return self._chunk_recipe_content(parsed_content)
        elif doc_type == 'menu_catalog':
            return self._chunk_menu_content(parsed_content)
        else:
            return self._chunk_generic_content(parsed_content)
    
    def _chunk_allergen_content(self, content: Dict[str, Any]) -> List[Dict]:
        """Create chunks for allergen documents - preserve table structure"""
        chunks = []
        
        # Extract allergen info
        allergen_info = content.get('allergen_info', {})
        
        # Create chunks for each allergen mentioned
        allergens = allergen_info.get('allergens_mentioned', [])
        for allergen in allergens:
            chunk = {
                'content': f"Allergène: {allergen}. " + content.get('text', '')[:500],
                'metadata': {
                    'allergen': allergen,
                    'chunk_type': 'allergen_info'
                },
                'document_type': 'allergen_table'
            }
            chunks.append(chunk)
        
        # If no specific allergens found, create a general chunk
        if not chunks:
            chunks.append({
                'content': content.get('text', '')[:1000],
                'metadata': {'chunk_type': 'allergen_general'},
                'document_type': 'allergen_table'
            })
        
        return chunks
    
    def _chunk_recipe_content(self, content: Dict[str, Any]) -> List[Dict]:
        """Create chunks for recipe documents"""
        chunks = []
        
        recipe_info = content.get('recipe_info', {})
        
        # Chunk for ingredients
        ingredients = recipe_info.get('ingredients', [])
        if ingredients:
            ingredients_text = f"Ingrédients: {', '.join(ingredients)}"
            chunks.append({
                'content': ingredients_text,
                'metadata': {'chunk_type': 'ingredients'},
                'document_type': 'recipe'
            })
        
        # Chunk for instructions
        instructions = recipe_info.get('instructions', [])
        if instructions:
            instructions_text = f"Instructions: {' '.join(instructions)}"
            chunks.append({
                'content': instructions_text,
                'metadata': {'chunk_type': 'instructions'},
                'document_type': 'recipe'
            })
        
        # General chunk with full content
        title = recipe_info.get('title', 'Recette')
        full_text = f"{title}. {content.get('text', '')}"
        chunks.append({
            'content': full_text[:1000],
            'metadata': {'chunk_type': 'full_recipe', 'title': title},
            'document_type': 'recipe'
        })
        
        return chunks
    
    def _chunk_menu_content(self, content: Dict[str, Any]) -> List[Dict]:
        """Create chunks for menu documents - one chunk per pizza"""
        chunks = []
        
        pizzas = content.get('pizzas', [])
        prices = content.get('prices', [])
        
        # Create chunks for each pizza
        for pizza in pizzas:
            pizza_name = pizza.get('name', '')
            chunk_content = f"Pizza: {pizza_name}. "
            
            # Try to find related text around the pizza name
            full_text = content.get('text', '')
            pizza_context = self._extract_pizza_context(full_text, pizza_name)
            if pizza_context:
                chunk_content += pizza_context
            
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'pizza_name': pizza_name,
                    'chunk_type': 'pizza_info'
                },
                'document_type': 'menu_catalog'
            })
        
        # If no pizzas found, create generic chunks
        if not chunks:
            chunks = self._chunk_generic_content(content)
        
        return chunks
    
    def _chunk_generic_content(self, content: Dict[str, Any]) -> List[Dict]:
        """Create generic text chunks"""
        text = content.get('text', '')
        chunks = []
        
        # Split text into chunks
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= self.chunk_size:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'metadata': {'chunk_type': 'generic'},
                    'document_type': content.get('document_type', 'unknown')
                })
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-self.chunk_overlap//10:]  # Approximate word overlap
                current_chunk = overlap_words
                current_size = sum(len(word) for word in overlap_words)
        
        # Add remaining words as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'metadata': {'chunk_type': 'generic'},
                'document_type': content.get('document_type', 'unknown')
            })
        
        return chunks
    
    def _extract_pizza_context(self, text: str, pizza_name: str) -> str:
        """Extract context around a pizza name"""
        # Find the position of the pizza name in text
        pizza_pos = text.lower().find(pizza_name.lower())
        if pizza_pos == -1:
            return ""
        
        # Extract surrounding context (e.g., 200 characters before and after)
        start = max(0, pizza_pos - 200)
        end = min(len(text), pizza_pos + len(pizza_name) + 200)
        
        context = text[start:end].strip()
        return context


