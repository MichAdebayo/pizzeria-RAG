# 🍕 Pizzeria RAG Assistant

A multi-agent Retrieval-Augmented Generation (RAG) system for pizzeria customer support, built with LangChain and Gradio.

## 📋 Overview

This system helps pizzeria customers get information about:
- Pizza menus and descriptions
- Ingredient lists and recipes  
- Allergen information (with safety protocols)
- Nutritional information
- Prices and availability

### Key Features

- **Multi-Agent Architecture**: Specialized agents for different query types
- **Advanced PDF Processing**: Handles text PDFs, scanned documents, and complex layouts
- **OCR Support**: Extracts text from image-based PDFs
- **Safety-First Allergen Checking**: High confidence thresholds for allergy-related queries
- **Local LLM Support**: Works with Ollama/local models
- **Fallback Systems**: Graceful degradation when components are unavailable

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** (for local LLM): [Install Ollama](https://ollama.ai)
3. **Tesseract OCR** (for scanned PDFs): 
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

### Installation

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd pizzeria-RAG
   pip install -r requirements.txt
   ```

2. **Start Ollama and download model**:
   ```bash
   ollama serve
   ollama pull llama3.2:latest
   ```

3. **Add your PDF documents**:
   ```bash
   # Place your pizza-related PDFs in:
   docs/raw_pdfs/
   ```

4. **Run setup**:
   ```bash
   python scripts/setup_environment.py
   ```

5. **Launch the interface**:
   ```bash
   python src/interface/gradio_app.py
   ```

## 📁 Project Structure

```
pizzeria-RAG/
├── config/                     # Configuration files
│   ├── settings.py            # Project settings
│   ├── agents_config.yaml     # LangChain agent configuration
│   └── extraction_config.yaml # Document processing configuration
├── docs/                      # Documents and data
│   ├── raw_pdfs/             # Input PDF files
│   ├── processed/            # Processed documents
│   └── validation/           # Quality validation results
├── src/
│   ├── core/                 # Core system components
│   │   ├── langchain_manager.py    # LangChain orchestration
│   │   ├── vector_store_manager.py # Vector store operations
│   │   └── document_processor.py   # Document processing pipeline
│   ├── extractors/           # PDF/document extraction
│   │   ├── pdf_extractor.py  # PDF text extraction
│   │   ├── ocr_extractor.py  # OCR for scanned documents
│   │   ├── table_extractor.py # Table extraction from PDFs
│   │   ├── recipe_extractor.py # Recipe information extraction
│   │   └── text_extractor.py # Plain text extraction
│   ├── parsers/              # Content parsing and classification
│   │   ├── document_parser.py # Document type detection
│   │   ├── pizza_parser.py   # Pizza information parsing
│   │   ├── allergen_parser.py # Allergen information parsing
│   │   └── recipe_parser.py  # Recipe parsing
│   ├── tools/                # LangChain tools (agents)
│   │   ├── pizza_search_tool.py     # Pizza information search
│   │   ├── allergen_check_tool.py   # Allergen safety checking
│   │   ├── ingredient_lookup_tool.py # Ingredient information
│   │   └── nutrition_info_tool.py   # Nutritional information
│   ├── utils/                # Utilities
│   │   ├── chunking.py       # Document chunking strategies
│   │   └── quality_assessment.py # Quality assessment
│   └── interface/
│       └── gradio_app.py     # Web interface
├── data/                     # Generated data
│   ├── vector_stores/        # Vector embeddings
│   └── processed_documents/  # Processed document cache
├── scripts/
│   └── setup_environment.py # Setup script
└── logs/                     # Application logs
```

## 🔧 Configuration

### Model Configuration

Edit `config/agents_config.yaml`:

```yaml
langchain_config:
  llm:
    provider: "ollama"
    model: "llama3.2:latest"  # Change model here
    temperature: 0.1
  
  embeddings:
    provider: "huggingface"
    model: "mixedbread-ai/mxbai-embed-large"
```

### Document Processing

Edit `config/extraction_config.yaml` to customize:
- OCR settings and languages
- Quality thresholds
- Chunking strategies
- Document classification rules

## 🔍 Usage Examples

### Web Interface

1. Start the Gradio app: `python src/interface/gradio_app.py`
2. Open http://localhost:7860
3. Ask questions like:
   - "Quelles pizzas avez-vous au menu?"
   - "La pizza Margherita contient-elle du gluten?"
   - "Quels sont les ingrédients de la pizza Quattro Stagioni?"

### Programmatic Usage

```python
from src.core.langchain_manager import LangChainManager

# Initialize the system
manager = LangChainManager('config/agents_config.yaml')

# Ask questions
response = manager.query("Avez-vous des pizzas sans lactose?")
print(response['response'])
```

## 🛠️ Document Processing Pipeline

The system processes documents through multiple stages:

1. **Extraction**: Multiple methods with fallbacks
   - `pdfplumber` for structured PDFs
   - `PyMuPDF` for complex layouts
   - `Tesseract OCR` for scanned documents
   - `EasyOCR` as OCR fallback

2. **Classification**: Automatic document type detection
   - Menu catalogs
   - Allergen tables
   - Recipe documents
   - Nutritional information

3. **Parsing**: Extract structured information
   - Pizza names and descriptions
   - Ingredient lists
   - Allergen matrices
   - Price information

4. **Chunking**: Create vector-store-ready chunks
   - Semantic chunking by content type
   - Preserve important relationships
   - Add relevant metadata

5. **Vector Storage**: Store in Chroma vector database
   - Separate stores by content type
   - Rich metadata for filtering
   - Optimized for retrieval

## 🔒 Safety Features

### Allergen Information
- **99% confidence threshold** for allergen-related responses
- **Safety fallback messages** when confidence is low
- **Source attribution** for all allergen information
- **Clear disclaimers** advising direct restaurant contact for severe allergies

### Error Handling
- **Graceful degradation** when components fail
- **Comprehensive logging** for debugging
- **Fallback responses** maintain system availability
- **Input validation** and sanitization

## 🚨 Troubleshooting

### Common Issues

1. **"LangChain not available"**
   ```bash
   pip install langchain langchain-community langchain-huggingface
   ```

2. **"Ollama connection failed"**
   ```bash
   ollama serve
   # In another terminal:
   ollama pull llama3.2:latest
   ```

3. **"OCR extraction failed"**
   ```bash
   # Install Tesseract
   brew install tesseract  # macOS
   sudo apt install tesseract-ocr  # Linux
   ```

4. **"No documents processed successfully"**
   - Check PDF files are in `docs/raw_pdfs/`
   - Verify files are readable (not password-protected)
   - Check logs in `logs/` directory

### Logs

Check these log files for detailed error information:
- `logs/setup.log` - Setup process
- `logs/extraction.log` - Document processing
- `logs/app.log` - Application runtime

## 🔄 Development

### Adding New Document Types

1. **Add classification patterns** in `config/extraction_config.yaml`
2. **Create parser** in `src/parsers/`
3. **Update chunking strategy** in `src/utils/chunking.py`
4. **Add vector store** in `config/agents_config.yaml`

### Adding New Tools

1. **Create tool class** in `src/tools/`
2. **Inherit from BaseToolMixin**
3. **Register in agents_config.yaml**
4. **Update tool imports** in `src/tools/__init__.py`

### Testing

```bash
# Test document processing
python scripts/setup_environment.py

# Test specific components
python -c "from src.extractors import PDFExtractor; print('Extractors working')"
python -c "from src.core.langchain_manager import LangChainManager; print('LangChain working')"
```

## 📊 Monitoring

### System Status
- Check vector store statistics
- Monitor extraction quality scores
- Review confidence thresholds
- Track response times

### Performance Tuning
- Adjust chunk sizes for your document types
- Optimize embedding model selection
- Configure OCR parameters for your PDF quality
- Tune LLM temperature and parameters

## 🤝 Contributing

1. **Document Processing**: Improve extraction for specific PDF layouts
2. **Agent Development**: Add specialized tools for specific queries
3. **Safety Features**: Enhance allergen detection and validation
4. **Performance**: Optimize vector search and chunking strategies
5. **UI/UX**: Improve the Gradio interface

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**⚠️ Important**: This system provides information assistance but should not replace direct communication with restaurant staff for critical allergen and safety information.