"""
Gradio interface for the Pizzeria RAG system
Clean web interface with document filtering and system management
"""

import gradio as gr
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.config import config
from src.core.rag_engine import LLMInterface
from src.core.pipeline import Pipeline

# Initialize components
pipeline = Pipeline()
llm_interface = LLMInterface()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_bot_response(history: List) -> List:
    """Get bot response for the last user message"""
    if not history or history[-1]["role"] != "user":
        return history
    
    question = history[-1]["content"]
    
    try:
        result = llm_interface.answer_question(question)
        
        if result['status'] == 'success':
            response = result['answer']
            
            # Log context information for debugging (not shown to user)
            if result['has_context']:
                if result['has_multiple_companies']:
                    companies = ", ".join(result['companies_found'])
                    logger.info(f"📊 Multi-company response - Sources: {companies}")
                else:
                    companies = ", ".join(result['companies_found']) if result['companies_found'] else "nos documents"
                    logger.info(f"📄 Single-company response - Source: {companies}")
            else:
                logger.info("⚠️ No specific context found in documents")
        else:
            response = f"❌ Désolé, une erreur s'est produite: {result.get('answer', 'Erreur inconnue')}"
            
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        response = f"❌ Erreur lors du traitement: {str(e)}"
    
    # Add assistant response
    history.append({"role": "assistant", "content": response})
    return history

def add_user_message(message: str, history: List) -> Tuple[str, List]:
    """Add user message to chat history and clear input"""
    if message.strip():
        history.append({"role": "user", "content": message})
    return "", history

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    .chat-container {
        height: 600px !important;
    }
    
    .input-container {
        max-width: 800px !important;
    }
    
    .status-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .header-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    
    .tab-nav {
        background-color: #34495e;
    }
    
    .tab-nav button {
        color: white !important;
    }
    
    .tab-nav button.selected {
        background-color: #3498db !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🍕 Pizzeria RAG Assistant", theme=gr.themes.Soft()) as interface: # type: ignore
        
        # Header
        gr.HTML("""
        <div class="header-title">
            <h1>🍕 Assistant Pizzeria RAG</h1>
            <p>Interface moderne pour rechercher dans nos menus de pizzerias</p>
        </div>
        """)
        
        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(
                    value=[],
                    elem_classes=["chat-container"],
                    height=600,
                    show_label=False,
                    type="messages"
                )
                
                with gr.Row():
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            placeholder="Posez votre question sur nos pizzerias... (ex: 'Quelles pizzas végétariennes avez-vous?')",
                            show_label=False,
                            max_lines=3,
                            elem_classes=["input-container"]
                        )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Envoyer", variant="primary", size="lg")
                
                gr.HTML("""
                <div style="margin-top: 15px; padding: 10px; background-color: #e8f4fd; border-radius: 8px;">
                    <h4>💡 Exemples de questions:</h4>
                    <ul>
                        <li>"Quelles sont vos spécialités?"</li>
                        <li>"Avez-vous des pizzas végétariennes?"</li>
                        <li>"Quel est le prix de la pizza Margherita?"</li>
                        <li>"Chez Marco Fuso, quels sont les horaires?"</li>
                    </ul>
                </div>
                """)
                        
                # Event handlers for chat
                submit_btn.click(
                    add_user_message,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot]
                ).then(
                    get_bot_response,
                    inputs=[chatbot],
                    outputs=[chatbot]
                )
                
                msg.submit(
                    add_user_message,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot]
                ).then(
                    get_bot_response,
                    inputs=[chatbot],
                    outputs=[chatbot]
                )
            
            # Help Tab
            with gr.TabItem("🆘 Aide"):
                gr.Markdown("""
                ## 🍕 Guide d'utilisation
                
                ### 💬 Comment poser des questions:
                1. **Questions générales**: "Quelles pizzas avez-vous?" → Recherche dans tous les documents
                2. **Questions spécifiques**: "Chez Anchor Pizza, quel est le prix de la Margherita?" → Filtre automatiquement
                3. **Comparaisons**: "Comparez les prix entre Marco Fuso et Anchor Pizza"
                
                ### 🔧 Fonctionnalités:
                - **Chat en temps réel** avec historique des conversations
                - **Recherche intelligente** dans tous les documents
                - **Filtrage automatique** par restaurant
                - **Gestion des documents** avec retraitement possible
                
                ### 💡 Conseils pour de meilleurs résultats:
                - Soyez spécifique dans vos questions
                - Mentionnez le nom du restaurant pour des résultats ciblés
                - Utilisez des mots-clés pertinents (prix, ingrédients, horaires, etc.)
                
                ### 🛠️ En cas de problème:
                1. Vérifiez que Ollama fonctionne: `ollama serve` dans le terminal
                2. Utilisez l'onglet "Documents" pour retraiter les documents si nécessaire
                3. Actualisez le statut du système pour vérifier les connexions
                
                ### 📞 Support:
                Si vous rencontrez des problèmes persistants, vérifiez les logs dans le terminal ou contactez l'administrateur système.
                """)
    
    return interface

def main():
    """Launch the Gradio interface"""
    logger.info("🍕 Starting Gradio Pizzeria RAG App")
    
    interface = create_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
