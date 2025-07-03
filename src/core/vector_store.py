"""
Modular vector store manager for the pizzeria RAG system
Based on the working simple_vector_store.py, now supporting multiple documents
"""

import json
import chromadb
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
import ollama
from config.config import config, DocumentConfig

class VectorStore:
    """Vector store manager for multiple documents using ChromaDB with Ollama embeddings"""
    
    def __init__(self):
        self.db_path = Path(config.vector_store.db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = config.models.embedding_model
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Store collections for each document
        self.collections = {}
        
        # Test Ollama connection
        self._test_ollama_connection()
    
    def _generate_ollama_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama (same as working simple version)"""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return response['embedding']
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1024
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available (same as working simple version)"""
        try:
            # Test embedding generation
            test_response = ollama.embeddings(model=self.embedding_model, prompt="test")
            self.logger.info(f"✅ Ollama embedding model '{self.embedding_model}' is ready")
        except Exception as e:
            self.logger.warning(f"⚠️ Ollama connection issue: {e}")
            self.logger.info("Make sure Ollama is running: ollama serve")
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]: # type: ignore
        """Split text into overlapping chunks (same as working simple version)"""
        chunk_size = chunk_size or config.vector_store.chunk_size
        overlap = overlap or config.vector_store.overlap
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def get_or_create_collection(self, document_name: str):
        """Get or create collection for a specific document"""
        if document_name in self.collections:
            return self.collections[document_name]
        
        collection_name = config.get_collection_name(document_name)
        
        try:
            # Try to get existing collection first
            collection = self.client.get_collection(name=collection_name)
            self.logger.info(f"Using existing collection: {collection_name}")
        except:
            # Collection doesn't exist, create new one
            try:
                doc_config = config.get_document_by_name(document_name)
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": doc_config.description,
                        "document_name": document_name,
                        "language": doc_config.language,
                        "content_type": doc_config.content_type
                    }
                )
                self.logger.info(f"Created new collection: {collection_name}")
            except Exception as e:
                self.logger.error(f"Error creating collection: {e}")
                raise Exception(f"Could not create or access collection: {e}")
        
        self.collections[document_name] = collection
        return collection
    
    def add_document(self, document_name: str) -> bool:
        """Add processed JSON document to vector store (based on working simple version)"""
        try:
            doc_config = config.get_document_by_name(document_name)
            json_path = doc_config.processed_json_path
            
            if not Path(json_path).exists():
                self.logger.error(f"Processed JSON not found: {json_path}")
                return False
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get collection for this document
            collection = self.get_or_create_collection(document_name)
            
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            # Process each section (same logic as working simple version)
            for i, section in enumerate(data.get('sections', [])):
                content = section.get('content', '')
                if not content:
                    continue
                
                # Chunk the content
                chunks = self.chunk_text(content)
                
                for j, chunk in enumerate(chunks):
                    doc_id = f"{document_name}_page_{section.get('page', i)}_chunk_{j}"
                    
                    # Generate embedding for this chunk
                    embedding = self._generate_ollama_embedding(chunk)
                    
                    documents.append(chunk)
                    embeddings.append(embedding)
                    metadatas.append({
                        "source": data.get('source', 'unknown'),
                        "document_name": document_name,
                        "page": section.get('page', i),
                        "chunk_index": j,
                        "word_count": len(chunk.split()),
                        "content_type": data.get('content_type', 'menu'),
                        "language": data.get('language', 'fr')
                    })
                    ids.append(doc_id)
            
            # Add to collection with manual embeddings
            if documents and embeddings:
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                self.logger.info(f"✅ Added {len(documents)} chunks for {document_name} with Ollama embeddings")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error adding document {document_name} to vector store: {e}")
            return False
    
    def search(self, query: str, document_names: Optional[Union[str, List[str]]] = None, 
               n_results: int = 5) -> Dict:
        """Search for relevant documents (enhanced from simple version to support multiple docs)"""
        try:
            # Generate embedding for the query
            query_embedding = self._generate_ollama_embedding(query)
            
            # Determine which documents to search
            if document_names is None:
                # Search all documents
                search_documents = [doc.name for doc in config.documents]
            elif isinstance(document_names, str):
                search_documents = [document_names]
            else:
                search_documents = document_names
            
            all_results = []
            
            # Search each document collection
            for doc_name in search_documents:
                try:
                    collection = self.get_or_create_collection(doc_name)
                    
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results
                    )
                    
                    # Format results for this document
                    if results['documents'] and results['documents'][0]: # type: ignore
                        for i, doc in enumerate(results['documents'][0]): # type: ignore
                            result = {
                                "content": doc,
                                "metadata": results['metadatas'][0][i] if results['metadatas'] else {}, # type: ignore
                                "distance": results['distances'][0][i] if results['distances'] else None, # type: ignore
                                "document_name": doc_name
                            }
                            all_results.append(result)
                
                except Exception as e:
                    self.logger.warning(f"Error searching document {doc_name}: {e}")
                    continue
            
            # Sort all results by distance and limit
            all_results.sort(key=lambda x: x.get('distance', float('inf')))
            all_results = all_results[:n_results]
            
            return {
                "query": query,
                "searched_documents": search_documents,
                "results": all_results
            }
            
        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            return {"query": query, "searched_documents": [], "results": []}
    
    def get_stats(self) -> Dict:
        """Get statistics for all collections"""
        stats = {
            "total_collections": 0,
            "total_documents": 0,
            "collections": {}
        }
        
        try:
            for doc_name in [doc.name for doc in config.documents]:
                try:
                    collection = self.get_or_create_collection(doc_name)
                    count = collection.count()
                    stats["collections"][doc_name] = {
                        "document_count": count,
                        "collection_name": collection.name
                    }
                    stats["total_documents"] += count
                    stats["total_collections"] += 1
                except Exception as e:
                    self.logger.warning(f"Error getting stats for {doc_name}: {e}")
                    stats["collections"][doc_name] = {"document_count": 0, "error": str(e)}
        
        except Exception as e:
            self.logger.error(f"Error getting overall stats: {e}")
        
        return stats
    
    def add_all_documents(self) -> Dict[str, bool]:
        """Add all configured documents to vector store"""
        results = {}
        
        for doc_config in config.documents:
            self.logger.info(f"Adding {doc_config.name} to vector store...")
            results[doc_config.name] = self.add_document(doc_config.name)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        self.logger.info(f"📊 Vector store update complete: {successful}/{total} documents successful")
        
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    vector_store = VectorStore()
    
    # Add all documents
    results = vector_store.add_all_documents()
    
    # Test search on all documents
    search_results = vector_store.search("pizza margherita")
    print(f"Search results: {len(search_results['results'])} found across {len(search_results['searched_documents'])} documents")
    
    # Get stats
    stats = vector_store.get_stats()
    print(f"Vector store stats: {stats}")
