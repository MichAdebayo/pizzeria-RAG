#!/usr/bin/env python3
"""
Setup script for the pizzeria RAG system
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from src.core.document_processor import DocumentProcessor
from src.core.vector_store_manager import VectorStoreManager
from src.core.langchain_manager import LangChainManager


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Settings.LOGS_DIR / 'setup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main setup function"""
    print("🍕 Setting up Pizzeria RAG System...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Create directories
        print("📁 Creating directories...")
        Settings.ensure_directories()
        logger.info("✓ Directories created")
        
        # Step 2: Check if documents exist
        if not Settings.RAW_PDFS_DIR.exists() or not list(Settings.RAW_PDFS_DIR.glob("*.pdf")):
            print(f"⚠️  No PDF files found in {Settings.RAW_PDFS_DIR}")
            print("   Please add PDF files to the raw_pdfs directory before running setup.")
            return
        
        pdf_count = len(list(Settings.RAW_PDFS_DIR.glob("*.pdf")))
        print(f"📄 Found {pdf_count} PDF files to process")
        
        # Step 3: Process documents
        print("🔄 Processing documents...")
        processor = DocumentProcessor(str(Settings.EXTRACTION_CONFIG))
        results = processor.process_all_documents(Settings.RAW_PDFS_DIR)
        
        successful_results = [r for r in results if r.get('status') == 'success']
        print(f"✓ Processed {len(successful_results)}/{len(results)} documents successfully")
        
        if not successful_results:
            print("❌ No documents were processed successfully")
            print("   Check the logs for detailed error information")
            return
        
        # Step 4: Initialize LangChain components
        print("🔗 Initializing LangChain components...")
        try:
            langchain_manager = LangChainManager(str(Settings.AGENTS_CONFIG))
            logger.info("✓ LangChain manager initialized")
            
            # Step 5: Create vector stores
            print("📊 Creating vector stores...")
            vector_manager = langchain_manager.vector_store_manager
            vector_manager.populate_stores_from_processed_docs(successful_results)
            
            # Get vector store statistics
            stats = vector_manager.get_store_stats()
            print("✓ Vector stores created:")
            for store_name, store_stats in stats.items():
                status = store_stats.get('status', 'unknown')
                doc_count = store_stats.get('document_count', 'unknown')
                print(f"   - {store_name}: {status} ({doc_count} documents)")
            
        except Exception as e:
            logger.error(f"LangChain initialization failed: {e}")
            print(f"⚠️  LangChain setup failed: {e}")
            print("   The system will run with fallback components")
        
        # Step 6: Test the system
        print("🧪 Testing system...")
        try:
            test_response = langchain_manager.query("Bonjour, quelles pizzas avez-vous?")
            if test_response.get('status') == 'success':
                print("✓ System test successful")
                print(f"   Test response: {test_response.get('response', '')[:100]}...")
            else:
                print(f"⚠️  System test failed: {test_response.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"⚠️  System test error: {e}")
        
        # Step 7: Show summary
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Run the Gradio interface:")
        print(f"   cd {project_root}")
        print("   python src/interface/gradio_app.py")
        print("\n2. Or test the system programmatically:")
        print("   python -c \"from src.core.langchain_manager import LangChainManager; lm = LangChainManager('config/agents_config.yaml'); print(lm.query('Quelles pizzas avez-vous?'))\"")
        
        print(f"\n📂 Files created:")
        print(f"   - Logs: {Settings.LOGS_DIR}")
        print(f"   - Processed documents: {Settings.PROCESSED_DIR}")
        print(f"   - Vector stores: {Settings.VECTOR_STORES_DIR}")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("Check the logs for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
