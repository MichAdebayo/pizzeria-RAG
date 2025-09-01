import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import fitz  # PyMuPDF
from config.config import config, DocumentConfig

class DocumentProcessor:
    """
    Handles the extraction and processing of documents for the pizzeria RAG system.
    Provides methods to extract text from PDFs, process documents into structured data, and manage batch processing.

    This class supports single and batch document processing, saving results in structured JSON format,
    and integrates with the system configuration for document management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text content from each page of a PDF file and returns structured sections.
        Returns a list of dictionaries containing page number, content, and word count for each page.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            List[Dict[str, Any]]: A list of section dictionaries with page, content, and word count.
        """
        try:
            doc = fitz.open(pdf_path)
            sections = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text() # type: ignore
                
                if text.strip():  # Only add pages with content
                    sections.append({
                        "page": page_num + 1,
                        "content": text.strip(),
                        "word_count": len(text.split())
                    })
            
            doc.close()
            self.logger.info(f"Extracted text from {len(sections)} pages")
            return sections
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return []
    
    def process_document(self, document_config: DocumentConfig) -> bool:
        """
        Processes a single document by extracting text from its PDF and saving structured data to JSON.
        Returns True if the document is successfully processed and saved, otherwise returns False.

        Args:
            document_config (DocumentConfig): The configuration object for the document to process.

        Returns:
            bool: True if the document is processed and saved successfully, False otherwise.
        """
        try:
            pdf_path = Path(document_config.pdf_path)
            if not pdf_path.exists():
                self.logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            self.logger.info(f"Processing document: {document_config.name}")
            
            # Extract text from PDF
            sections = self.extract_text_from_pdf(str(pdf_path))
            
            if not sections:
                self.logger.error(f"No content extracted from {pdf_path}")
                return False
            
            # Create structured data
            structured_data = {
                "source": str(pdf_path),
                "document_name": document_config.name,
                "description": document_config.description,
                "language": document_config.language,
                "content_type": document_config.content_type,
                "total_pages": len(sections),
                "total_words": sum(section["word_count"] for section in sections),
                "sections": sections,
                "processing_metadata": {
                    "processor_version": "modular_v1.0",
                    "extraction_method": "PyMuPDF"
                }
            }
            
            # Save processed data
            output_path = Path(document_config.processed_json_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Processed {document_config.name}: {len(sections)} pages, "
                           f"{structured_data['total_words']} words")
            self.logger.info(f"📄 Saved to: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing document {document_config.name}: {e}")
            return False
    
    def process_all_documents(self) -> Dict[str, bool]:
        """
        Processes all documents defined in the system configuration and saves their structured data.
        Returns a dictionary mapping document names to their processing success status.

        Returns:
            Dict[str, bool]: A dictionary with document names as keys and processing success as values.
        """
        results = {}
        
        for doc_config in config.documents:
            self.logger.info(f"Processing {doc_config.name}...")
            results[doc_config.name] = self.process_document(doc_config)
        
        # Summary
        successful = sum(bool(success)
                     for success in results.values())
        total = len(results)
        self.logger.info(f"📊 Processing complete: {successful}/{total} documents successful")
        
        return results
    
    def process_document_by_name(self, document_name: str) -> bool:
        """
        Processes a single document by its name using the system configuration.
        Returns True if the document is successfully processed, otherwise returns False.

        Args:
            document_name (str): The name of the document to process.

        Returns:
            bool: True if the document is processed successfully, False otherwise.
        """
        try:
            doc_config = config.get_document_by_name(document_name)
            return self.process_document(doc_config)
        except ValueError as e:
            self.logger.error(f"Document not found: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = DocumentProcessor()
    
    # Process all documents
    results = processor.process_all_documents()
    
    # Print results
    for doc_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {doc_name}")
