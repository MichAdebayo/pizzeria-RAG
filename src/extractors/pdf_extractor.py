from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .base_extractor import BaseExtractor


class PDFExtractor(BaseExtractor):
    """Extractor for PDF files using multiple methods"""
    
    def __init__(self, method: str = 'pdfplumber', config: Dict = None):
        super().__init__(config)
        self.method = method
        self.supported_formats = ['.pdf']
        
        # Try to import PDF libraries
        self.available_methods = {}
        self._setup_extractors()
    
    def _setup_extractors(self):
        """Setup available PDF extraction methods"""
        try:
            import pdfplumber
            self.available_methods['pdfplumber'] = pdfplumber
        except ImportError:
            self.logger.warning("pdfplumber not available")
        
        try:
            import fitz  # PyMuPDF
            self.available_methods['pymupdf'] = fitz
        except ImportError:
            self.logger.warning("PyMuPDF not available")
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the given file"""
        return (
            file_path.suffix.lower() in self.supported_formats and
            len(self.available_methods) > 0
        )
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from PDF file"""
        if not self.can_handle(file_path):
            raise ValueError(f"Cannot handle file: {file_path}")
        
        try:
            if self.method == 'pdfplumber' and 'pdfplumber' in self.available_methods:
                return self._extract_with_pdfplumber(file_path)
            elif self.method == 'pymupdf' and 'pymupdf' in self.available_methods:
                return self._extract_with_pymupdf(file_path)
            else:
                # Fallback to first available method
                available_method = list(self.available_methods.keys())[0]
                self.logger.info(f"Using fallback method: {available_method}")
                return getattr(self, f'_extract_with_{available_method}')(file_path)
                
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return {
                'text': '',
                'metadata': self._get_basic_metadata(file_path),
                'error': str(e),
                'confidence': 0.0
            }
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Dict[str, Any]:
        """Extract using pdfplumber"""
        import pdfplumber
        
        text_content = []
        tables = []
        metadata = self._get_basic_metadata(file_path)
        
        with pdfplumber.open(file_path) as pdf:
            metadata['page_count'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                page_text = page.extract_text() or ""
                text_content.append({
                    'page': page_num,
                    'text': page_text
                })
                
                # Extract tables
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend([{
                        'page': page_num,
                        'table_index': i,
                        'data': table
                    } for i, table in enumerate(page_tables)])
        
        all_text = "\n\n".join([page['text'] for page in text_content if page['text']])
        
        return {
            'text': all_text,
            'pages': text_content,
            'tables': tables,
            'metadata': metadata,
            'confidence': self._calculate_confidence(all_text)
        }
    
    def _extract_with_pymupdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract using PyMuPDF"""
        import fitz
        
        text_content = []
        metadata = self._get_basic_metadata(file_path)
        
        doc = fitz.open(file_path)
        metadata['page_count'] = len(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            text_content.append({
                'page': page_num + 1,
                'text': page_text
            })
        
        doc.close()
        
        all_text = "\n\n".join([page['text'] for page in text_content if page['text']])
        
        return {
            'text': all_text,
            'pages': text_content,
            'metadata': metadata,
            'confidence': self._calculate_confidence(all_text)
        }
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate extraction confidence based on text quality"""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Basic heuristics for text quality
        score = 0.5  # Base score
        
        # Check for reasonable word count
        words = text.split()
        if len(words) > 20:
            score += 0.2
        
        # Check for reasonable character variety
        if len(set(text.lower())) > 10:
            score += 0.2
        
        # Check for pizza-related terms
        pizza_terms = ['pizza', 'ingrédient', 'allergen', 'gluten', 'lactose', 'fromage']
        found_terms = sum(1 for term in pizza_terms if term.lower() in text.lower())
        if found_terms > 0:
            score += min(0.3, found_terms * 0.1)
        
        return min(1.0, score)
    
    def validate_extraction(self, result: Dict[str, Any]) -> float:
        """Return confidence score for extraction quality"""
        return result.get('confidence', 0.0)
