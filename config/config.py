"""
Configuration for the modular pizzeria RAG system
Based on the working simple_* implementation, now generalized for multiple documents
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for Ollama models"""
    chat_model: str = "llama3.2:latest"
    embedding_model: str = "mxbai-embed-large"
    temperature: float = 0.3
    top_p: float = 0.9
    num_predict: int = 300

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    db_path: str = "data/vector_db"
    chunk_size: int = 500
    overlap: int = 50
    collection_prefix: str = "pizzeria"

@dataclass
class DocumentConfig:
    """Configuration for a single document"""
    name: str
    pdf_path: str
    processed_json_path: str
    description: str
    language: str = "fr"
    content_type: str = "menu"

class ModularConfig:
    """Main configuration class for the modular RAG system"""
    
    def __init__(self):
        self.models = ModelConfig()
        self.vector_store = VectorStoreConfig()
        
        # Paths (set these first)
        self.base_path = Path(".")
        self.docs_path = self.base_path / "docs"
        self.raw_pdfs_path = self.docs_path / "raw_pdfs"
        self.processed_path = self.base_path / "data" / "processed"
        self.logs_path = self.base_path / "logs"
        
        # Create directories
        self._create_directories()
        
        # Initialize documents (after paths are set)
        self.documents = self._init_documents()
    
    def _init_documents(self) -> List[DocumentConfig]:
        """Initialize document configurations by auto-discovering PDFs in raw_pdfs directory"""
        documents = []
        
        # Create the raw_pdfs directory if it doesn't exist
        self.raw_pdfs_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files in the raw_pdfs directory
        pdf_files = list(self.raw_pdfs_path.glob("*.pdf")) + list(self.raw_pdfs_path.glob("*.PDF"))
        
        for pdf_path in pdf_files:
            # Extract clean name from filename
            raw_name = pdf_path.stem  # Get filename without extension
            
            # Clean the name: remove hyphens, underscores, make title case
            clean_name = raw_name.replace('-', ' ').replace('_', ' ').title()
            
            # Create document name (lowercase with underscores for internal use)
            doc_name = raw_name.lower().replace('-', '_').replace(' ', '_')
            
            # Create document configuration
            doc_config = DocumentConfig(
                name=doc_name,
                pdf_path=str(pdf_path),
                processed_json_path=f"data/processed/{doc_name}_processed.json",
                description=f"{clean_name} - Pizzeria menu and services",
                language="fr",
                content_type="menu"
            )
            
            documents.append(doc_config)
        
        # Log discovered documents
        if documents:
            print(f"📁 Auto-discovered {len(documents)} PDF documents:")
            for doc in documents:
                print(f"   - {doc.name}: {doc.description}")
        else:
            print("⚠️  No PDF files found in docs/raw_pdfs/")
            print(f"   Please add PDF files to: {self.raw_pdfs_path}")
        
        return documents
    
    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.raw_pdfs_path, self.processed_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_document_by_name(self, name: str) -> DocumentConfig:
        """Get document configuration by name"""
        for doc in self.documents:
            if doc.name == name:
                return doc
        raise ValueError(f"Document '{name}' not found")
    
    def get_collection_name(self, document_name: str) -> str:
        """Get ChromaDB collection name for a document"""
        return f"{self.vector_store.collection_prefix}_{document_name}_ollama"
    
    def add_document(self, name: str, pdf_path: str, description: str, 
                    language: str = "fr", content_type: str = "menu"):
        """Add a new document to the configuration"""
        processed_json_path = f"data/processed/{name}_processed.json"
        
        new_doc = DocumentConfig(
            name=name,
            pdf_path=pdf_path,
            processed_json_path=processed_json_path,
            description=description,
            language=language,
            content_type=content_type
        )
        
        self.documents.append(new_doc)
        return new_doc
    
    def get_available_documents(self) -> Dict[str, str]:
        """Get a dictionary of available documents (name -> description)"""
        return {doc.name: doc.description for doc in self.documents}
    
    def get_company_name(self, document_name: str) -> str:
        """Get clean company name from document name"""
        try:
            doc_config = self.get_document_by_name(document_name)
            # Extract company name from description (everything before the first hyphen)
            return doc_config.description.split(' - ')[0]
        except:
            # Fallback: clean the document name
            return document_name.replace('_', ' ').title()
    
    def get_companies_summary(self) -> Dict[str, Dict]:
        """Get summary of all companies/documents"""
        companies = {}
        for doc in self.documents:
            company_name = self.get_company_name(doc.name)
            companies[doc.name] = {
                'company_name': company_name,
                'description': doc.description,
                'pdf_path': doc.pdf_path,
                'language': doc.language
            }
        return companies

# Global configuration instance
config = ModularConfig()
