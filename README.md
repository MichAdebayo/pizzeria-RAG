# 🍕 Pizzeria RAG - Multi-Document Search Assistant

<div align="center">
  <img src="assets/pizza_asset.webp" alt="Pizzeria RAG System" width="800">
</div>

A modern RAG (Retrieval-Augmented Generation) system for querying multiple pizzeria menus with Ollama and ChromaDB.

## 🚀 Features

- **Multi-Document RAG**: Simultaneous search across multiple pizzeria menus
- **🚨 Smart Allergen Detection**: Comprehensive allergen analysis and safety warnings
- **Modern Interface**: Chainlit and Gradio for optimal user experience
- **Local-First**: Uses Ollama for embeddings and LLM (no cloud dependency)
- **Modular Architecture**: Clean, maintainable and extensible code
- **Intelligent Processing**: Automatic PDF extraction and indexing

## 🛠️ Architecture

```
pizzeria-RAG/
├── assets/                  # Images and resources
│   └── pizza_asset.webp
├── config/                  # Centralized configuration
│   └── config.py
├── data/                    # Processed data and vector database
│   ├── processed/           # Structured document JSONs
│   └── vector_db/           # ChromaDB vector database (git-ignored)
├── docs/                    # Documentation and sources (git-ignored)
│   └── raw_pdfs/            # Source PDF documents
├── logs/                    # Log files (git-ignored)
├── processors/              # Document processing
│   ├── __init__.py
│   └── document_processor.py
├── src/                     # Main source code
│   ├── __init__.py
│   ├── apps/                # User applications
│   │   ├── __init__.py
│   │   ├── chainlit_app.py      # Chainlit chat interface
│   │   └── gradio_app.py        # Gradio web interface
│   └── core/                # Core RAG logic
│       ├── pipeline.py          # Main pipeline
│       ├── rag_engine.py        # RAG engine (allergen logic)
│       └── vector_store.py      # Embeddings & ChromaDB management
│    
├── chainlit.md              # Chainlit configuration
├── requirements.txt         # Python dependencies
├── start_chainlit.sh        # Chainlit launch script
├── start_gradio.sh          # Gradio launch script
├── .gitignore               # Git ignored files
└── README.md                # Project documentation
```

## 📋 Prerequisites

1. **Ollama** installed and configured
2. **Python 3.9+**
3. **Ollama Models**:
   - `llama3.2:latest` (chat)
   - `mxbai-embed-large` (embeddings)

## 🔧 Installation

1. **Clone the project**:
```bash
git clone <repository-url>
cd pizzeria-RAG
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Start Ollama** (in a separate terminal):
```bash
ollama serve
```

5. **Download models**:
```bash
ollama pull llama3.2:latest
ollama pull mxbai-embed-large
```

## 🎯 Usage

### Chainlit Interface (Recommended)
```bash
./start_chainlit.sh
```
**Access:** http://localhost:8000

**Features:**
- Modern interactive chat
- Integrated system commands (`/status`, `/documents`, `/process`, `/help`)
- Real-time system status display
- Clean interface without technical clutter

### Gradio Interface (Alternative)
```bash
./start_gradio.sh
```
**Access:** http://localhost:7860

**Features:**
- Simple and intuitive web interface
- Chat with user/assistant avatars
- Integrated help tab
- Perfect for demos and quick testing

### Direct Python Usage
```python
from src.core.rag_engine import LLMInterface

llm = LLMInterface()
result = llm.answer_question("What vegetarian pizzas do you have?")
print(result['answer'])
```

## 💬 Example Questions

- **General**: "What pizzas do you have?"
- **Specific**: "At Anchor Pizza, what's the price of the Margherita?"
- **Comparative**: "Compare prices between Marco Fuso and Anchor Pizza"
- **Detailed**: "What are the ingredients in the Veggie Triumph?"
- **🚨 Allergen-Related**: "I'm allergic to gluten and dairy, what can I eat?"
- **Safety**: "Do you have pizzas without nuts?"
- **Multiple Allergens**: "I need to avoid gluten, eggs, and shellfish"

## 🔧 System Commands (Chainlit)

- `/status` - Detailed system status
- `/documents` - List available documents
- `/process` - Process/reprocess documents
- `/help` - Complete help

## 📁 Data

The system automatically processes PDFs in `docs/raw_pdfs/`:
- **Anchor Pizza**: Menu and services
- **Marco Fuso**: Recipes and techniques

Processed data is stored in:
- `data/processed/` - Structured JSONs
- `data/vector_db/` - ChromaDB vector database

## 🚨 Allergen Detection System

The system includes comprehensive allergen detection and safety features:

### 📋 Supported Allergens (EU Standards)
- **Gluten** (wheat, rye, barley, oats, spelt, kamut)
- **Crustaceans** (shrimp, lobster, crab, langoustine)
- **Eggs** (albumin, lecithin)
- **Fish** (anchovy, tuna, salmon, sardine)
- **Peanuts** (groundnuts, peanut butter)
- **Soy** (soy sauce, soy lecithin, soy protein)
- **Milk** (lactose, casein, whey, cheese, cream)
- **Tree Nuts** (almonds, hazelnuts, walnuts, pistachios)
- **Celery** (celery root, celery seed)
- **Mustard** (mustard seeds, mustard powder)
- **Sesame** (sesame seeds, sesame oil, tahini)
- **Sulfites** (sulfur dioxide, preservatives E220-E228)
- **Lupin** (lupin flour, lupin protein)
- **Mollusks** (oysters, mussels, scallops, squid)

### 🔍 Smart Detection Features
- **Automatic User Allergen Detection**: Extracts allergens from user questions
- **Ingredient Analysis**: Scans menu items for allergen presence
- **Multi-Pattern Recognition**: Handles various expression formats
  - "I'm allergic to gluten and dairy"
  - "No nuts please"
  - "Avoiding shellfish"
  - "Lactose intolerant"
- **Safety Warnings**: Always displays allergen information, even when not asked
- **Alternative Suggestions**: Recommends suitable options based on restrictions

### 🛡️ Safety First Approach
- **Always Visible**: Allergen info included in every response
- **Clear Warnings**: Uses emoji indicators (⚠️ for warnings, ✅ for safe)
- **Multi-Restaurant Analysis**: Compares allergen safety across different menus
- **Comprehensive Coverage**: Analyzes ingredients, sauces, and preparation methods

## ⚙️ Configuration

Modify `config/config.py` to adjust:
- Ollama models used
- Chunking parameters
- LLM temperature
- Ports and endpoints

## 🐛 Troubleshooting

### Common issues:

1. **Ollama not connected**:
   ```bash
   ollama serve
   curl http://localhost:11434/api/tags
   ```

2. **Missing models**:
   ```bash
   ollama list
   ollama pull llama3.2:latest
   ollama pull mxbai-embed-large
   ```

3. **Launch scripts**:
   ```bash
   # Make scripts executable
   chmod +x start_chainlit.sh start_gradio.sh
   
   # Check virtual environment
   source .venv/bin/activate
   ```

4. **Documents not indexed**:
   - Use `/process` in Chainlit
   - Or check `data/vector_db/`

5. **Performance issues**:
   - Reduce `chunk_size` in config
   - Check available RAM

6. **Gradio errors**:
   ```bash
   # Update Gradio if necessary
   pip install --upgrade gradio
   ```

## 🚀 Development

### Recent improvements:
- **Clean Chainlit interface**: Status system moved to system commands
- **Clean user responses**: Removal of technical information from responses
- **🚨 Comprehensive Allergen Detection**: 14 major allergens with smart detection
- **Safety-First Approach**: Always-visible allergen warnings and recommendations
- **Refactored architecture**: Clear code organization in modules
- **Avatar management**: Modern user interfaces with avatars
- **Launch scripts**: Simplified startup with automatic checks

### Code structure:
- **Modular**: Each component has its responsibility
- **Type Hints**: Typed code for better maintenance
- **Logging**: Complete operation traceability
- **Async/Await**: Non-blocking interface for better UX
- **Error handling**: Robustness and automatic recovery

### Adding new documents:
1. Place PDF in `docs/raw_pdfs/`
2. Add configuration in `config/config.py`
3. Restart processing via `/process` (Chainlit) or Gradio interface

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

In case of issues:
1. Check the Troubleshooting section
2. Check application logs
3. Open an issue on GitHub

## 📄 License

This project is under MIT license. See `LICENSE` for more details.