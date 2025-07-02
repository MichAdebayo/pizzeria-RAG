from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .base_extractor import BaseExtractor


class OCRExtractor(BaseExtractor):
    """Extractor for scanned PDFs and images using OCR"""
    
    def __init__(self, method: str = 'tesseract', config: Dict = None):
        super().__init__(config or {})
        self.method = method
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff']
        
        # Setup OCR engines
        self.available_engines = {}
        self._setup_ocr_engines()
    
    def _setup_ocr_engines(self):
        """Setup available OCR engines"""
        try:
            import pytesseract
            self.available_engines['tesseract'] = pytesseract
        except ImportError:
            self.logger.warning("Tesseract not available")
        
        try:
            import easyocr
            self.available_engines['easyocr'] = easyocr
        except ImportError:
            self.logger.warning("EasyOCR not available")
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the given file"""
        return (
            file_path.suffix.lower() in self.supported_formats and
            len(self.available_engines) > 0
        )
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content using OCR"""
        if not self.can_handle(file_path):
            raise ValueError(f"Cannot handle file: {file_path}")
        
        try:
            # First convert PDF to images if needed
            images = self._prepare_images(file_path)
            
            # Run OCR on images
            text_results = []
            for i, image in enumerate(images):
                page_text = self._run_ocr(image)
                text_results.append({
                    'page': i + 1,
                    'text': page_text
                })
            
            all_text = "\n\n".join([page['text'] for page in text_results if page['text']])
            
            return {
                'text': all_text,
                'pages': text_results,
                'metadata': self._get_basic_metadata(file_path),
                'confidence': self._calculate_ocr_confidence(all_text)
            }
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return {
                'text': '',
                'metadata': self._get_basic_metadata(file_path),
                'error': str(e),
                'confidence': 0.0
            }
    
    def _prepare_images(self, file_path: Path) -> List:
        """Convert PDF to images or load image files"""
        images = []
        
        if file_path.suffix.lower() == '.pdf':
            # Convert PDF to images
            try:
                import pdf2image
                images = pdf2image.convert_from_path(file_path, dpi=300)
            except ImportError:
                self.logger.error("pdf2image not available for PDF conversion")
                # Fallback: try to read as single image
                try:
                    from PIL import Image
                    images = [Image.open(file_path)]
                except Exception:
                    self.logger.error("Cannot convert PDF to images")
        else:
            # Load image file directly
            try:
                from PIL import Image
                images = [Image.open(file_path)]
            except Exception as e:
                self.logger.error(f"Cannot load image: {e}")
        
        return images
    
    def _run_ocr(self, image) -> str:
        """Run OCR on a single image"""
        if self.method == 'tesseract' and 'tesseract' in self.available_engines:
            return self._run_tesseract(image)
        elif self.method == 'easyocr' and 'easyocr' in self.available_engines:
            return self._run_easyocr(image)
        else:
            # Fallback to any available engine
            if self.available_engines:
                engine_name = list(self.available_engines.keys())[0]
                return getattr(self, f'_run_{engine_name}')(image)
            return ""
    
    def _run_tesseract(self, image) -> str:
        """Run Tesseract OCR"""
        try:
            import pytesseract
            
            # Configure for French/English
            config = '--oem 3 --psm 6 -l fra+eng'
            return pytesseract.image_to_string(image, config=config)
        except Exception as e:
            self.logger.error(f"Tesseract OCR failed: {e}")
            return ""
    
    def _run_easyocr(self, image) -> str:
        """Run EasyOCR"""
        try:
            import easyocr
            import numpy as np
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['fr', 'en'])
            results = reader.readtext(img_array)
            
            # Extract text from results
            text = " ".join([result[1] for result in results])
            return text
        except Exception as e:
            self.logger.error(f"EasyOCR failed: {e}")
            return ""
    
    def _calculate_ocr_confidence(self, text: str) -> float:
        """Calculate OCR confidence based on text quality"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        # Basic confidence calculation
        score = 0.3  # Base score for OCR
        
        # Check for reasonable word count
        words = text.split()
        if len(words) > 10:
            score += 0.2
        
        # Check for special characters (common OCR errors)
        special_chars = sum(1 for char in text if not char.isalnum() and char not in ' .,!?-\n')
        if special_chars < len(text) * 0.1:  # Less than 10% special chars
            score += 0.3
        
        # Check for pizza-related terms
        pizza_terms = ['pizza', 'ingredient', 'allergen', 'gluten', 'lactose']
        found_terms = sum(1 for term in pizza_terms if term.lower() in text.lower())
        if found_terms > 0:
            score += min(0.2, found_terms * 0.05)
        
        return min(1.0, score)
    
    def validate_extraction(self, result: Dict[str, Any]) -> float:
        """Return confidence score for extraction quality"""
        return result.get('confidence', 0.0)
