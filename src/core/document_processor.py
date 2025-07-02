from pathlib import Path
from typing import List, Dict, Any
import logging
import yaml

from src.extractors import PDFExtractor, OCRExtractor, TextExtractor
from src.parsers.document_parser import DocumentParser
from src.parsers.pizza_parser import PizzaParser
from src.parsers.allergen_parser import AllergenParser
from src.parsers.recipe_parser import RecipeParser
from src.utils.chunking import DocumentChunker, QualityAssessment


class DocumentProcessor:
    """Main document processing pipeline"""
    
    def __init__(self, extraction_config_path: str):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(extraction_config_path)
        
        # Initialize components
        self.extractors = self._setup_extractors()
        self.classifier = DocumentParser(self.config)
        self.parsers = self._setup_parsers()
        self.quality_assessor = QualityAssessment(self.config)
        self.chunker = DocumentChunker(self.config.get('chunking', {}))
    
    def _load_config(self, config_path: str) -> Dict:
        """Load extraction configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for document processing"""
        return {
            'pdf_extraction': {
                'extraction_methods': [
                    {'name': 'pdfplumber', 'priority': 1},
                    {'name': 'pymupdf', 'priority': 2},
                    {'name': 'tesseract_ocr', 'priority': 3}
                ]
            },
            'document_processing': {
                'quality_thresholds': {
                    'min_extraction_confidence': 0.8
                }
            }
        }
    
    def _setup_extractors(self):
        """Initialize extraction methods"""
        return {
            'pdfplumber': PDFExtractor('pdfplumber'),
            'pymupdf': PDFExtractor('pymupdf'),
            'tesseract_ocr': OCRExtractor('tesseract'),
            'easyocr': OCRExtractor('easyocr'),
            'text': TextExtractor()
        }
    
    def _setup_parsers(self):
        """Initialize content parsers"""
        return {
            'menu_catalog': PizzaParser(self.config),
            'allergen_table': AllergenParser(self.config),
            'recipe': RecipeParser(self.config),
            'nutrition': RecipeParser(self.config)  # Reuse recipe parser for nutrition
        }
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document through the complete pipeline"""
        
        self.logger.info(f"Processing document: {file_path.name}")
        
        # Step 1: Extract content with fallbacks
        extracted_content = self._extract_with_fallbacks(file_path)
        if not extracted_content or not extracted_content.get('text'):
            return {
                'status': 'extraction_failed', 
                'file': str(file_path),
                'error': 'No content extracted'
            }
        
        # Step 2: Classify document type
        doc_type = self.classifier.classify(extracted_content)
        self.logger.info(f"Document classified as: {doc_type}")
        
        # Step 3: Parse content based on type
        parsed_content = self._parse_content(extracted_content, doc_type)
        
        # Step 4: Validate quality
        quality_score = self.quality_assessor.assess(parsed_content)
        self.logger.info(f"Quality score: {quality_score:.2f}")
        
        # Step 5: Create chunks for vector storage
        chunks = self.chunker.create_chunks(parsed_content, doc_type)
        
        return {
            'status': 'success',
            'file': str(file_path),
            'document_type': doc_type,
            'extracted_content': extracted_content,
            'parsed_content': parsed_content,
            'quality_score': quality_score,
            'chunks': chunks,
            'metadata': {
                'extraction_method': extracted_content.get('method', 'unknown'),
                'processing_timestamp': self._get_timestamp(),
                'confidence': extracted_content.get('confidence', 0.0),
                'chunk_count': len(chunks)
            }
        }
    
    def _extract_with_fallbacks(self, file_path: Path) -> Dict[str, Any]:
        """Try extraction methods in priority order"""
        extraction_methods = self.config.get('pdf_extraction', {}).get('extraction_methods', [])
        
        # Sort by priority
        extraction_methods = sorted(extraction_methods, key=lambda x: x.get('priority', 999))
        
        for method_config in extraction_methods:
            method_name = method_config.get('name')
            extractor = self.extractors.get(method_name)
            
            if not extractor:
                self.logger.warning(f"Extractor {method_name} not available")
                continue
                
            try:
                self.logger.info(f"Trying extraction method: {method_name}")
                result = extractor.extract(file_path)
                
                if self._validate_extraction(result):
                    result['method'] = method_name
                    self.logger.info(f"Successfully extracted with {method_name}")
                    return result
                else:
                    self.logger.warning(f"Extraction validation failed for {method_name}")
                    
            except Exception as e:
                self.logger.warning(f"Extraction method {method_name} failed: {e}")
                continue
        
        self.logger.error("All extraction methods failed")
        return None
    
    def _validate_extraction(self, result: Dict) -> bool:
        """Basic validation of extraction result"""
        if not result or 'text' not in result:
            return False
        
        text = result['text']
        if not text or len(text.strip()) < 10:  # Minimum text threshold
            return False
        
        # Check confidence if available
        confidence = result.get('confidence', 0.0)
        min_confidence = self.config.get('document_processing', {}).get('quality_thresholds', {}).get('min_extraction_confidence', 0.5)
        
        if confidence < min_confidence:
            self.logger.warning(f"Extraction confidence {confidence:.2f} below threshold {min_confidence}")
            # Don't fail completely, but log the warning
        
        return True
    
    def _parse_content(self, content: Dict, doc_type: str) -> Dict[str, Any]:
        """Parse content based on document type"""
        parser = self.parsers.get(doc_type)
        if parser:
            try:
                parsed = parser.parse(content)
                # Ensure we include the original text
                parsed['text'] = content.get('text', '')
                parsed['document_type'] = doc_type
                return parsed
            except Exception as e:
                self.logger.error(f"Parsing failed for {doc_type}: {e}")
        
        # Fallback: return content as-is with document type
        content['document_type'] = doc_type
        return content
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def process_all_documents(self, docs_dir: Path) -> List[Dict]:
        """Process all documents in directory"""
        results = []
        
        if not docs_dir.exists():
            self.logger.error(f"Directory does not exist: {docs_dir}")
            return results
        
        # Process PDF files
        pdf_files = list(docs_dir.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing {pdf_file.name}")
                result = self.process_document(pdf_file)
                results.append(result)
                
                # Log result summary
                if result.get('status') == 'success':
                    self.logger.info(f"✓ {pdf_file.name}: {result.get('document_type')} - Quality: {result.get('quality_score', 0):.2f}")
                else:
                    self.logger.error(f"✗ {pdf_file.name}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file.name}: {e}")
                results.append({
                    'status': 'error',
                    'file': str(pdf_file),
                    'error': str(e)
                })
        
        # Summary
        successful = sum(1 for r in results if r.get('status') == 'success')
        self.logger.info(f"Processing complete: {successful}/{len(results)} documents processed successfully")
        
        return results