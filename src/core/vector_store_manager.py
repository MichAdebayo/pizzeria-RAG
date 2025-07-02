from typing import Dict, List, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path
import logging
import json
import yaml

class VectorStoreManager:
    """Manages all vector stores for the pizzeria RAG system"""
    
    def __init__(self, config: Dict = None, embeddings=None):
        self.config = config or {}
        self.embeddings = embeddings
        self.vector_stores = {}
        self.logger = logging.getLogger(__name__)
        
        if config and embeddings:
            self._initialize_stores()
    
    def _initialize_stores(self):
        """Initialize all vector stores from config"""
        vector_store_configs = self.config.get('langchain_config', {}).get('vector_stores', {})
        
        for store_name, store_config in vector_store_configs.items():
            try:
                if store_config['type'] == 'chroma':
                    persist_dir = Path(store_config['persist_directory'])
                    persist_dir.mkdir(parents=True, exist_ok=True)
                    
                    self.vector_stores[store_name] = Chroma(
                        persist_directory=str(persist_dir),
                        embedding_function=self.embeddings,
                        collection_name=store_config['collection_name']
                    )
                    self.logger.info(f"Initialized vector store: {store_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize vector store {store_name}: {e}")
    
    def get_store(self, store_name: str) -> Optional[Chroma]:
        """Get a specific vector store"""
        return self.vector_stores.get(store_name)
    
    def get_all_stores(self) -> Dict[str, Chroma]:
        """Get all vector stores"""
        return self.vector_stores
    
    def add_documents_to_store(self, store_name: str, documents: List[Document]):
        """Add documents to a specific vector store"""
        store = self.get_store(store_name)
        if store:
            try:
                store.add_documents(documents)
                self.logger.info(f"Added {len(documents)} documents to {store_name}")
            except Exception as e:
                self.logger.error(f"Failed to add documents to {store_name}: {e}")
        else:
            self.logger.error(f"Vector store {store_name} not found")
    
    def create_documents_from_chunks(self, chunks: List[Dict], source_file: str) -> List[Document]:
        """Convert processed chunks to LangChain documents"""
        documents = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                'source': source_file,
                'chunk_id': i,
                'page_number': chunk.get('page_number', 0),
                'document_type': chunk.get('document_type', 'unknown'),
                'confidence': chunk.get('confidence', 0.0)
            }
            
            # Add any additional metadata from the chunk
            if 'metadata' in chunk:
                metadata.update(chunk['metadata'])
            
            doc = Document(
                page_content=chunk.get('content', ''),
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def populate_stores_from_processed_docs(self, processed_docs: List[Dict]):
        """Populate vector stores from processed documents"""
        for doc_result in processed_docs:
            if doc_result.get('status') != 'success':
                continue
                
            file_name = doc_result.get('file', '')
            doc_type = doc_result.get('document_type', 'unknown')
            chunks = doc_result.get('chunks', [])
            
            if not chunks:
                self.logger.warning(f"No chunks found for {file_name}")
                continue
            
            # Convert chunks to documents
            documents = self.create_documents_from_chunks(chunks, file_name)
            
            # Route to appropriate vector store based on document type
            store_name = self._get_store_for_doc_type(doc_type)
            if store_name:
                self.add_documents_to_store(store_name, documents)
            else:
                # Default to pizza_descriptions store
                self.add_documents_to_store('pizza_descriptions', documents)
    
    def _get_store_for_doc_type(self, doc_type: str) -> Optional[str]:
        """Map document type to vector store"""
        mapping = {
            'menu_catalog': 'pizza_descriptions',
            'allergen_table': 'allergen_data',
            'recipe': 'recipes',
            'nutrition': 'nutrition_data'
        }
        return mapping.get(doc_type)
    
    def search_store(self, store_name: str, query: str, k: int = 5) -> List[Document]:
        """Search a specific vector store"""
        store = self.get_store(store_name)
        if store:
            try:
                return store.similarity_search(query, k=k)
            except Exception as e:
                self.logger.error(f"Search failed in {store_name}: {e}")
                return []
        return []
    
    def get_store_stats(self) -> Dict[str, Dict]:
        """Get statistics for all vector stores"""
        stats = {}
        for name, store in self.vector_stores.items():
            try:
                # Try to get collection info
                collection = store._collection
                stats[name] = {
                    'name': name,
                    'type': 'chroma',
                    'document_count': collection.count() if hasattr(collection, 'count') else 'unknown',
                    'status': 'active'
                }
            except Exception as e:
                stats[name] = {
                    'name': name,
                    'type': 'chroma', 
                    'status': 'error',
                    'error': str(e)
                }
        return stats
