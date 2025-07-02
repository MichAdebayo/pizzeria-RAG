from typing import Dict, List, Any, Optional
from pathlib import Path
import re
import logging
from .base_extractor import BaseExtractor


class RecipeExtractor(BaseExtractor):
    """Specialized extractor for recipe information from documents"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.supported_formats = ['.pdf', '.txt', '.docx']
        
        # Recipe detection patterns
        self.recipe_patterns = self.config.get('recipe_extraction', {
            'recipe_indicators': [
                'recette', 'recipe', 'ingrédients', 'ingredients', 
                'préparation', 'preparation', 'étapes', 'steps'
            ],
            'ingredient_patterns': [
                r'(\d+(?:[.,]\d+)?\s*(?:g|gr|grammes?|kg|kilogrammes?|ml|cl|l|litres?|c\.?\s*à\s*s\.?|cuillères?\s*à\s*soupe|c\.?\s*à\s*c\.?|cuillères?\s*à\s*café|pincée|poignée|tranches?|morceaux?|unités?|pièces?))\s*([a-zA-ZÀ-ÿ\s]+)',
                r'([a-zA-ZÀ-ÿ\s]+)\s*[:]\s*(\d+(?:[.,]\d+)?\s*(?:g|gr|grammes?|kg|ml|cl|l|c\.?\s*à\s*s\.?|c\.?\s*à\s*c\.?))',
                r'[-•]\s*([a-zA-ZÀ-ÿ\s]+?)(?:\s*[:]\s*(\d+(?:[.,]\d+)?\s*(?:g|gr|grammes?|kg|ml|cl|l|c\.?\s*à\s*s\.?|c\.?\s*à\s*c\.?)))?'
            ],
            'step_patterns': [
                r'(\d+)[.)]\s*([^.]+\.)',
                r'(Étape\s*\d+|Step\s*\d+)\s*[:]\s*([^.]+\.)',
                r'[-•]\s*([^.]+\.)'
            ]
        })
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file"""
        return file_path.suffix.lower() in self.supported_formats
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract recipe information from document"""
        if not self.can_handle(file_path):
            return self._empty_result(file_path, "Unsupported format")
        
        try:
            # Get text content (this would normally come from a text extractor)
            text_content = self._extract_text_content(file_path)
            
            if not text_content:
                return self._empty_result(file_path, "No text content found")
            
            # Extract recipe components
            recipes = self._extract_recipes(text_content)
            
            metadata = self._get_basic_metadata(file_path)
            metadata['extraction_method'] = 'recipe_parser'
            
            result = {
                'recipes': recipes,
                'recipe_count': len(recipes),
                'metadata': metadata,
                'confidence': self._calculate_confidence(recipes, text_content)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Recipe extraction failed: {e}")
            return self._empty_result(file_path, f"Extraction error: {str(e)}")
    
    def validate_extraction(self, result: Dict[str, Any]) -> float:
        """Validate recipe extraction quality"""
        recipes = result.get('recipes', [])
        
        if not recipes:
            return 0.0
        
        total_score = 0.0
        for recipe in recipes:
            score = 0.0
            
            # Check if recipe has name
            if recipe.get('name'):
                score += 0.2
            
            # Check if recipe has ingredients
            ingredients = recipe.get('ingredients', [])
            if ingredients:
                score += 0.4
                # Bonus for having quantities
                with_quantities = sum(1 for ing in ingredients if ing.get('quantity'))
                if with_quantities > 0:
                    score += 0.2 * (with_quantities / len(ingredients))
            
            # Check if recipe has steps
            steps = recipe.get('steps', [])
            if steps:
                score += 0.4
            
            total_score += score
        
        return min(1.0, total_score / len(recipes))
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract text content from file"""
        try:
            if file_path.suffix.lower() == '.txt':
                return file_path.read_text(encoding='utf-8')
            elif file_path.suffix.lower() == '.pdf':
                # This would normally use a PDF extractor
                # For now, return empty string as fallback
                self.logger.warning("PDF text extraction not implemented in recipe extractor")
                return ""
            else:
                return ""
        except Exception as e:
            self.logger.error(f"Failed to extract text from {file_path}: {e}")
            return ""
    
    def _extract_recipes(self, text: str) -> List[Dict[str, Any]]:
        """Extract recipe information from text"""
        recipes = []
        
        # Look for recipe sections
        recipe_sections = self._identify_recipe_sections(text)
        
        for section in recipe_sections:
            recipe = self._parse_recipe_section(section)
            if recipe and self._is_valid_recipe(recipe):
                recipes.append(recipe)
        
        # If no clear sections found, try to extract from entire text
        if not recipes:
            recipe = self._parse_recipe_section(text)
            if recipe and self._is_valid_recipe(recipe):
                recipes.append(recipe)
        
        return recipes
    
    def _identify_recipe_sections(self, text: str) -> List[str]:
        """Identify separate recipe sections in text"""
        sections = []
        
        # Split by common recipe separators
        separators = [
            r'(?i)^recette\s+(?:pour\s+)?(.+)$',
            r'(?i)^pizza\s+(.+)$',
            r'(?i)^recipe\s+for\s+(.+)$',
            r'(?i)^\d+\.\s*(.+)$'
        ]
        
        current_section = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts a new recipe
            is_new_recipe = any(re.match(pattern, line) for pattern in separators)
            
            if is_new_recipe and current_section:
                # Save previous section
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections if sections else [text]
    
    def _parse_recipe_section(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a single recipe section"""
        recipe = {
            'name': self._extract_recipe_name(text),
            'ingredients': self._extract_ingredients(text),
            'steps': self._extract_preparation_steps(text),
            'metadata': {
                'servings': self._extract_servings(text),
                'cooking_time': self._extract_cooking_time(text),
                'difficulty': self._extract_difficulty(text)
            }
        }
        
        return recipe
    
    def _extract_recipe_name(self, text: str) -> str:
        """Extract recipe name from text"""
        lines = text.split('\n')
        
        # Look for title patterns
        title_patterns = [
            r'(?i)^(?:recette\s+(?:pour\s+)?|pizza\s+|recipe\s+for\s+)?(.+)$',
            r'(?i)^(.+)(?:\s+recipe|\s+recette)$'
        ]
        
        for line in lines[:3]:  # Check first few lines
            line = line.strip()
            if len(line) > 5 and len(line) < 50:  # Reasonable title length
                for pattern in title_patterns:
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1).strip()
                        if not any(keyword in name.lower() for keyword in ['ingrédient', 'ingredient', 'étape', 'step']):
                            return name
        
        return "Recette sans nom"
    
    def _extract_ingredients(self, text: str) -> List[Dict[str, Any]]:
        """Extract ingredients list from text"""
        ingredients = []
        
        # Find ingredients section
        ingredient_section = self._find_section(text, ['ingrédients', 'ingredients'])
        
        if ingredient_section:
            for pattern in self.recipe_patterns['ingredient_patterns']:
                matches = re.findall(pattern, ingredient_section, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    if isinstance(match, tuple):
                        if len(match) == 2:
                            # Pattern: quantity + ingredient or ingredient + quantity
                            quantity, ingredient = match
                            if re.search(r'\d', quantity):
                                ingredients.append({
                                    'name': ingredient.strip(),
                                    'quantity': quantity.strip()
                                })
                            else:
                                ingredients.append({
                                    'name': quantity.strip(),
                                    'quantity': ingredient.strip() if ingredient else None
                                })
                    else:
                        # Simple ingredient without quantity
                        ingredients.append({
                            'name': match.strip(),
                            'quantity': None
                        })
        
        # Remove duplicates and clean up
        seen = set()
        cleaned_ingredients = []
        for ing in ingredients:
            name = ing['name'].lower().strip()
            if name and name not in seen and len(name) > 2:
                seen.add(name)
                cleaned_ingredients.append(ing)
        
        return cleaned_ingredients
    
    def _extract_preparation_steps(self, text: str) -> List[str]:
        """Extract preparation steps from text"""
        steps = []
        
        # Find preparation section
        prep_section = self._find_section(text, ['préparation', 'preparation', 'étapes', 'steps', 'méthode', 'method'])
        
        if prep_section:
            for pattern in self.recipe_patterns['step_patterns']:
                matches = re.findall(pattern, prep_section, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    if isinstance(match, tuple):
                        step_text = match[1] if len(match) > 1 else match[0]
                    else:
                        step_text = match
                    
                    step_text = step_text.strip()
                    if len(step_text) > 10:  # Filter out very short steps
                        steps.append(step_text)
        
        return steps
    
    def _find_section(self, text: str, section_keywords: List[str]) -> Optional[str]:
        """Find a specific section in the text"""
        lines = text.split('\n')
        section_start = -1
        
        # Find section start
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in section_keywords):
                section_start = i
                break
        
        if section_start == -1:
            return None
        
        # Find section end (next major section or end of text)
        section_end = len(lines)
        major_sections = ['ingrédients', 'ingredients', 'préparation', 'preparation', 'étapes', 'steps']
        
        for i in range(section_start + 1, len(lines)):
            line = lines[i].lower().strip()
            if any(section in line for section in major_sections):
                if not any(keyword in line for keyword in section_keywords):
                    section_end = i
                    break
        
        return '\n'.join(lines[section_start:section_end])
    
    def _extract_servings(self, text: str) -> Optional[str]:
        """Extract serving information"""
        patterns = [
            r'(?i)(?:pour|serves?|portions?)\s*[:]?\s*(\d+(?:\s*[-à]\s*\d+)?)',
            r'(?i)(\d+)\s*(?:personnes?|portions?|servings?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_cooking_time(self, text: str) -> Optional[str]:
        """Extract cooking time information"""
        patterns = [
            r'(?i)(?:cuisson|cooking|baking)\s*[:]\s*(\d+(?:\s*h\s*\d+)?\s*(?:min|minutes?|h|heures?))',
            r'(?i)(\d+(?:\s*h\s*\d+)?\s*(?:min|minutes?|h|heures?))\s*(?:de\s*)?(?:cuisson|cooking|baking)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_difficulty(self, text: str) -> Optional[str]:
        """Extract difficulty level"""
        difficulty_keywords = {
            'facile': ['facile', 'easy', 'simple'],
            'moyen': ['moyen', 'medium', 'intermediate'],
            'difficile': ['difficile', 'hard', 'difficult', 'advanced']
        }
        
        text_lower = text.lower()
        for level, keywords in difficulty_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        return None
    
    def _is_valid_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Check if extracted recipe is valid"""
        # Must have either ingredients or steps
        has_ingredients = bool(recipe.get('ingredients'))
        has_steps = bool(recipe.get('steps'))
        
        return has_ingredients or has_steps
    
    def _calculate_confidence(self, recipes: List[Dict], text: str) -> float:
        """Calculate confidence score for recipe extraction"""
        if not recipes:
            return 0.0
        
        score = 0.0
        
        # Check for recipe indicators in text
        indicators = self.recipe_patterns['recipe_indicators']
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in text.lower())
        score += min(0.3, indicator_count * 0.1)
        
        # Check recipe quality
        for recipe in recipes:
            recipe_score = 0.0
            
            if recipe.get('name') and recipe['name'] != "Recette sans nom":
                recipe_score += 0.2
            
            if recipe.get('ingredients'):
                recipe_score += 0.4
            
            if recipe.get('steps'):
                recipe_score += 0.4
            
            score += recipe_score / len(recipes)
        
        return min(1.0, score)
    
    def _empty_result(self, file_path: Path, reason: str) -> Dict[str, Any]:
        """Return empty result with error information"""
        return {
            'recipes': [],
            'recipe_count': 0,
            'metadata': {
                **self._get_basic_metadata(file_path),
                'error': reason,
                'extraction_method': 'recipe_parser'
            },
            'confidence': 0.0
        }
