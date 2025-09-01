from dataclasses import dataclass
from typing import List, Dict, Any, Optional
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

@dataclass
class AllergenConfig:
    """Configuration for allergen detection and management"""
    allergens_list: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.allergens_list is None:
            self.allergens_list = [
                "Gluten",
                "Crustacés", 
                "Oeufs",
                "Poissons",
                "Arachides",
                "Soja",
                "Lait",
                "Fruits à coque",
                "Céleri",
                "Moutarde",
                "Sésame",
                "Sulfites",
                "Lupin",
                "Mollusques"
            ]
    
    def get_allergen_keywords(self) -> Dict[str, List[str]]:
        """Get keywords to detect each allergen in ingredients"""
        return {
            "Gluten": ["gluten", "blé", "froment", "épeautre", "kamut", "seigle", "orge", "avoine", "farine"],
            "Crustacés": ["crustacés", "crustacé", "crevette", "homard", "crabe", "langoustine", "écrevisse"],
            "Oeufs": ["oeuf", "œuf", "oeufs", "œufs", "albumine", "lecithine d'oeuf"],
            "Poissons": ["poisson", "poissons", "anchois", "thon", "saumon", "sardine", "colin", "cabillaud"],
            "Arachides": ["arachide", "arachides", "cacahuète", "cacahouète", "beurre de cacahuète"],
            "Soja": ["soja", "sauce soja", "lecithine de soja", "protéine de soja"],
            "Lait": ["lait", "crème", "beurre", "fromage", "yaourt", "lactose", "caséine", "lactosérum"],
            "Fruits à coque": ["fruits à coque", "amande", "noisette", "noix", "pistache", "cajou", "pécan", "macadamia", "pignon"],
            "Céleri": ["céleri", "céleri-rave"],
            "Moutarde": ["moutarde", "graines de moutarde"],
            "Sésame": ["sésame", "graines de sésame", "huile de sésame", "tahini"],
            "Sulfites": ["sulfite", "sulfites", "anhydride sulfureux", "E220", "E221", "E222", "E223", "E224", "E225", "E226", "E227", "E228"],
            "Lupin": ["lupin", "farine de lupin"],
            "Mollusques": ["mollusque", "mollusques", "escargot", "huître", "moule", "coquille saint-jacques", "calmar", "poulpe"]
        }

class ModularConfig:
    """Main configuration class for the modular RAG system"""
    
    def __init__(self):
        self.models = ModelConfig()
        self.vector_store = VectorStoreConfig()
        self.allergen = AllergenConfig()
        
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
        """
        Initializes the list of document configurations by scanning the raw_pdfs directory for PDF files.
        Returns a list of DocumentConfig objects for each discovered PDF, creating names and descriptions automatically.

        Returns:
            List[DocumentConfig]: A list of configuration objects for all discovered documents.
        """
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
        """
        Creates necessary directories for raw PDFs, processed data, and logs if they do not already exist.
        Ensures the required folder structure is present for document management and system operation.

        Returns:
            None
        """
        for path in [self.raw_pdfs_path, self.processed_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_document_by_name(self, name: str) -> DocumentConfig:
        """
        Retrieves the DocumentConfig object for a document by its internal name.
        Returns the configuration object if found, otherwise raises a ValueError.

        Args:
            name (str): The internal name of the document to retrieve.

        Returns:
            DocumentConfig: The configuration object for the specified document.

        Raises:
            ValueError: If the document with the given name is not found.
        """
        for doc in self.documents:
            if doc.name == name:
                return doc
        raise ValueError(f"Document '{name}' not found")
    
    def get_collection_name(self, document_name: str) -> str:
        """
        Generates the collection name for a document in the vector store using the configured prefix.
        Returns the formatted collection name as a string.

        Args:
            document_name (str): The name of the document for which to generate the collection name.

        Returns:
            str: The formatted collection name for the vector store.
        """
        return f"{self.vector_store.collection_prefix}_{document_name}_ollama"
    
    def add_document(self, name: str, pdf_path: str, description: str, 
                    language: str = "fr", content_type: str = "menu"):
        """
        Adds a new document to the configuration with the specified details.
        Returns the newly created DocumentConfig object after appending it to the documents list.

        Args:
            name (str): The internal name for the document.
            pdf_path (str): The file path to the PDF document.
            description (str): A description of the document.
            language (str, optional): The language of the document. Defaults to "fr".
            content_type (str, optional): The type of content. Defaults to "menu".

        Returns:
            DocumentConfig: The configuration object for the newly added document.
        """
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
        """
        Retrieves a dictionary of available documents with their names and descriptions.
        Returns a mapping of document names to their descriptions for use in selection or display.

        Returns:
            Dict[str, str]: A dictionary with document names as keys and descriptions as values.
        """
        return {doc.name: doc.description for doc in self.documents}
    
    def get_company_name(self, document_name: str) -> str:
        """
        Retrieves the company name associated with a document by extracting it from the document's description.
        Returns the company name as a string, or a cleaned version of the document name if extraction fails.

        Args:
            document_name (str): The name of the document to retrieve the company name for.

        Returns:
            str: The extracted company name or a cleaned document name as fallback.
        """
        try:
            doc_config = self.get_document_by_name(document_name)
            # Extract company name from description (everything before the first hyphen)
            return doc_config.description.split(' - ')[0]
        except Exception:
            # Fallback: clean the document name
            return document_name.replace('_', ' ').title()
    
    def get_companies_summary(self) -> Dict[str, Dict]:
        """
        Generates a summary of all companies represented by the configured documents.
        Returns a dictionary mapping document names to company details including name, description, PDF path, and language.

        Returns:
            Dict[str, Dict]: A dictionary with document names as keys and company details as values.
        """
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
