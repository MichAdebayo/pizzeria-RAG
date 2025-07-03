"""
Main pipeline for the modular pizzeria RAG system
Process documents and manage the vector store
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from processors.document_processor import DocumentProcessor
from src.core.vector_store import VectorStore  
from src.core.rag_engine import LLMInterface

class Pipeline:
    """Main pipeline for processing documents and managing the RAG system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_interface = LLMInterface()
    
    def process_all_documents(self) -> bool:
        """Process all documents through the complete pipeline"""
        self.logger.info("🚀 Starting complete pipeline for all documents...")
        
        # Step 1: Process PDFs to JSON
        self.logger.info("📄 Step 1: Processing PDFs to structured JSON...")
        processing_results = self.processor.process_all_documents()
        
        # Step 2: Add to vector store
        self.logger.info("🔍 Step 2: Adding documents to vector store...")
        vector_results = self.vector_store.add_all_documents()
        
        # Summary
        total_docs = len(config.documents)
        processed_success = sum(1 for success in processing_results.values() if success)
        vector_success = sum(1 for success in vector_results.values() if success)
        
        self.logger.info(f"📊 Pipeline Summary:")
        self.logger.info(f"   - PDF Processing: {processed_success}/{total_docs} successful")
        self.logger.info(f"   - Vector Store: {vector_success}/{total_docs} successful")
        
        return processed_success == total_docs and vector_success == total_docs
    
    def process_single_document(self, document_name: str) -> bool:
        """Process a single document through the complete pipeline"""
        self.logger.info(f"🚀 Starting pipeline for document: {document_name}")
        
        try:
            # Validate document exists
            doc_config = config.get_document_by_name(document_name)
            
            # Step 1: Process PDF to JSON
            self.logger.info(f"📄 Step 1: Processing {document_name} PDF to JSON...")
            processing_success = self.processor.process_document_by_name(document_name)
            
            if not processing_success:
                self.logger.error(f"❌ Failed to process {document_name}")
                return False
            
            # Step 2: Add to vector store
            self.logger.info(f"🔍 Step 2: Adding {document_name} to vector store...")
            vector_success = self.vector_store.add_document(document_name)
            
            if not vector_success:
                self.logger.error(f"❌ Failed to add {document_name} to vector store")
                return False
            
            self.logger.info(f"✅ Pipeline complete for {document_name}")
            return True
            
        except ValueError as e:
            self.logger.error(f"❌ Document not found: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Pipeline error for {document_name}: {e}")
            return False
    
    def test_system(self, document_name: Optional[str] = None) -> Dict:
        """Test the system with sample queries"""
        self.logger.info("🧪 Testing system...")
        
        test_questions = [
            "Quelles pizzas avez-vous au menu?",
            "Quel est le prix de la pizza Margherita?",
            "Avez-vous des pizzas végétariennes?"
        ]
        
        results = []
        
        for question in test_questions:
            self.logger.info(f"Testing: {question}")
            result = self.llm_interface.answer_question(question, document_names=document_name)
            results.append(result)
            
            # Log result
            status = "✅" if result['status'] == 'success' else "❌"
            context_status = "📄" if result['has_context'] else "📭"
            self.logger.info(f"{status} {context_status} {question}")
        
        return {
            "total_tests": len(test_questions),
            "successful_tests": sum(1 for r in results if r['status'] == 'success'),
            "tests_with_context": sum(1 for r in results if r['has_context']),
            "results": results
        }
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return self.llm_interface.get_system_status()
    
    def add_new_document(self, name: str, pdf_path: str, description: str) -> bool:
        """Add a new document to the system"""
        try:
            # Add to configuration
            doc_config = config.add_document(name, pdf_path, description)
            self.logger.info(f"📝 Added new document config: {name}")
            
            # Process the document
            success = self.process_single_document(name)
            
            if success:
                self.logger.info(f"✅ Successfully added new document: {name}")
            else:
                self.logger.error(f"❌ Failed to add new document: {name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error adding new document {name}: {e}")
            return False

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Modular Pizzeria RAG Pipeline")
    parser.add_argument("--command", choices=["process-all", "process-single", "test", "status"], 
                       required=True, help="Command to execute")
    parser.add_argument("--document", help="Document name for single document operations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    pipeline = Pipeline()
    
    if args.command == "process-all":
        success = pipeline.process_all_documents()
        exit(0 if success else 1)
    
    elif args.command == "process-single":
        if not args.document:
            print("Error: --document required for process-single command")
            exit(1)
        success = pipeline.process_single_document(args.document)
        exit(0 if success else 1)
    
    elif args.command == "test":
        test_results = pipeline.test_system(args.document)
        print(f"Test Results: {test_results['successful_tests']}/{test_results['total_tests']} successful")
        print(f"Tests with context: {test_results['tests_with_context']}/{test_results['total_tests']}")
        exit(0 if test_results['successful_tests'] == test_results['total_tests'] else 1)
    
    elif args.command == "status":
        status = pipeline.get_system_status()
        print("System Status:")
        print(f"  Ollama Chat: {'✅' if status['ollama_chat'] else '❌'}")
        print(f"  Ollama Embeddings: {'✅' if status['ollama_embeddings'] else '❌'}")
        print(f"  Vector Store Collections: {status['vector_store'].get('total_collections', 0)}")
        print(f"  Total Documents in Vector Store: {status['vector_store'].get('total_documents', 0)}")
        print("  Document Status:")
        for doc_name, doc_status in status['documents'].items():
            pdf_status = "✅" if doc_status['pdf_exists'] else "❌"
            json_status = "✅" if doc_status['json_exists'] else "❌"
            print(f"    {doc_name}: PDF {pdf_status} | JSON {json_status} | {doc_status['description']}")

if __name__ == "__main__":
    main()
