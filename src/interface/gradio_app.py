import gradio as gr
import logging
import sys
from pathlib import Path

# Add the project root to the path to ensure imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings

# Try to import LangChain manager with fallback
try:
    from src.core.langchain_manager import LangChainManager
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain manager not available: {e}")
    LANGCHAIN_AVAILABLE = False

class PizzeriaRAGApp:
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, Settings.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize LangChain manager with fallback
        try:
            if LANGCHAIN_AVAILABLE:
                self.langchain_manager = LangChainManager(str(Settings.AGENTS_CONFIG))
                self.system_status = "✅ Système opérationnel"
            else:
                self.langchain_manager = None
                self.system_status = "⚠️ Mode développement (LangChain non disponible)"
        except Exception as e:
            logging.error(f"Failed to initialize LangChain manager: {e}")
            self.langchain_manager = None
            self.system_status = f"⚠️ Erreur d'initialisation: {str(e)[:100]}"
        
    def chat_response(self, message, history):
        """Handle chat interactions"""
        try:
            if self.langchain_manager:
                result = self.langchain_manager.query(message)
                
                if result['status'] == 'success':
                    return result['response']
                else:
                    return f"Désolé, une erreur s'est produite: {result.get('error', 'Erreur inconnue')}"
            else:
                # Fallback response when system is not fully operational
                return (f"🍕 Bonjour! Je suis l'assistant de la Pizzeria Bella Napoli.\n\n"
                       f"Vous avez demandé: \"{message}\"\n\n"
                       f"⚠️ Le système complet n'est pas encore opérationnel. "
                       f"Veuillez exécuter le script de configuration d'abord:\n"
                       f"```\npython scripts/setup_environment.py\n```\n\n"
                       f"Status: {self.system_status}")
                
        except Exception as e:
            logging.error(f"Chat error: {e}")
            return f"Désolé, une erreur s'est produite: {str(e)}"
    
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(
            title="Assistant Pizzeria Bella Napoli",
            theme=gr.themes.Soft()
        ) as demo:
            
            gr.Markdown("# 🍕 Assistant Pizzeria Bella Napoli")
            gr.Markdown("*Posez vos questions sur nos pizzas, ingrédients et allergènes!*")
            
            # Add system status
            gr.Markdown(f"**Status:** {self.system_status}")
            
            chatbot = gr.Chatbot(
                height=500,
                placeholder="Bonjour! Comment puis-je vous aider avec nos pizzas?",
                label="Assistant Pizzeria"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Tapez votre question ici...",
                    label="Votre question",
                    scale=4
                )
                submit_btn = gr.Button("Envoyer", scale=1, variant="primary")
            
            clear_btn = gr.Button("Nouvelle conversation", variant="secondary")
            
            # Event handlers
            msg.submit(self.chat_response, [msg, chatbot], [chatbot])
            submit_btn.click(self.chat_response, [msg, chatbot], [chatbot])
            clear_btn.click(lambda: None, None, chatbot, queue=False)
            
            # Examples
            gr.Examples(
                examples=[
                    "Quelles pizzas avez-vous au menu?",
                    "La pizza Margherita contient-elle du gluten?",
                    "Quels sont les ingrédients de la pizza Quattro Stagioni?",
                    "Avez-vous des pizzas sans lactose?"
                ],
                inputs=msg
            )
        
        return demo

def main():
    """Main application entry point"""
    # Ensure directories exist
    Settings.ensure_directories()
    
    # Create and launch app
    app = PizzeriaRAGApp()
    demo = app.create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )

if __name__ == "__main__":
    main()