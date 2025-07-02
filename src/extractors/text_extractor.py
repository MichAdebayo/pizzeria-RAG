from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .base_extractor import BaseExtractor


class TextExtractor(BaseExtractor):
    """Simple extractor for plain text files"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.supported_formats = ['.txt', '.md', '.csv']
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the given file"""
        return file_path.suffix.lower() in self.supported_formats
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from text file"""
        if not self.can_handle(file_path):
            raise ValueError(f"Cannot handle file: {file_path}")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            text = ""
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if not text:
                raise ValueError("Could not decode file with any encoding")
            
            return {
                'text': text,
                'pages': [{'page': 1, 'text': text}],
                'metadata': self._get_basic_metadata(file_path),
                'confidence': 1.0  # Perfect confidence for text files
            }
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return {
                'text': '',
                'metadata': self._get_basic_metadata(file_path),
                'error': str(e),
                'confidence': 0.0
            }
    
    def validate_extraction(self, result: Dict[str, Any]) -> float:
        """Return confidence score for extraction quality"""
        return result.get('confidence', 0.0)
