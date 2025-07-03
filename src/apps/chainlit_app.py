"""
Chainlit interface for the Pizzeria RAG system
Modern chat interface with document filtering and system management
"""

import chainlit as cl
import logging
import asyncio
from typing import List, Optional, Dict
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from src.core.rag_engine import LLMInterface
from src.core.pipeline import Pipeline

# Initialize components
pipeline = Pipeline()
llm_interface = LLMInterface()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def store_session_data(status, available_docs):
    """Store status data in session for easy access"""
    cl.user_session.set("system_status", status)
    cl.user_session.set("available_docs", available_docs)
    logger.info("📊 System status and document info stored in session") 


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    logger.info("🍕 Starting new Chainlit session")
    
    # Get system status asynchronously
    loop = asyncio.get_event_loop()
    status = await loop.run_in_executor(None, llm_interface.get_system_status)
    available_docs = await loop.run_in_executor(None, llm_interface.get_available_documents)
    
    # Store status data in session
    await store_session_data(status, available_docs)
    
    # Create welcome message (clean, without system status)
    welcome_msg = f"""# 🍕 Bienvenue dans l'Assistant Pizzeria!

## 💬 Comment utiliser:
1. **Questions générales**: "Quelles pizzas avez-vous?" → Recherche dans tous les documents
2. **Questions spécifiques**: "Chez Anchor Pizza, quel est le prix de la Margherita?" → Filtre automatiquement
3. **Commandes système**: Tapez `/status` pour voir l'état du système

**Exemples de questions:**
- "Quelles sont vos spécialités?"
- "Avez-vous des pizzas végétariennes?"
- "Quel est le prix de la pizza Margherita?"
- "Chez Marco Fuso, quels sont les horaires?"

**Commandes disponibles:**
- **`/status`** - Voir le statut détaillé du système 📊
- `/documents` - Lister les documents
- `/process` - Traiter les documents
- `/help` - Aide détaillée

💡 **Astuce**: Utilisez `/status` pour vérifier que le système est opérationnel!

Posez votre question et je vous aiderai! 🚀
"""
    
    await cl.Message(content=welcome_msg).send()
    
    # Store session data
    cl.user_session.set("available_documents", available_docs)
    cl.user_session.set("system_status", status)

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    user_message = message.content.strip()
    
    if not user_message:
        await cl.Message(content="Veuillez poser une question sur nos pizzerias! 🍕").send()
        return
    
    # Handle special commands
    if user_message.startswith('/'):
        await handle_command(user_message)
        return
    
    # Send thinking message and process
    async with cl.Step(name="search", type="run") as step:
        step.output = "🤔 Je cherche dans nos documents..."
        
        try:
            # Process the question in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, llm_interface.answer_question, user_message)
            
            if result['status'] == 'success':
                response_content = result['answer']
                
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
                
                step.output = "✅ Recherche terminée"
                
            else:
                response_content = f"❌ Désolé, une erreur s'est produite: {result.get('answer', 'Erreur inconnue')}"
                step.output = "❌ Erreur lors de la recherche"
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            step.output = f"❌ Erreur: {str(e)}"
            response_content = f"❌ Erreur lors du traitement: {str(e)}"
    
    # Send the final response
    await cl.Message(content=response_content).send()

async def handle_command(command: str):
    """Handle special commands"""
    
    if command == '/status':
        await show_status()
    
    elif command == '/documents':
        await show_documents()
    
    elif command == '/process':
        await process_documents()
    
    elif command == '/help':
        await show_help()
    
    else:
        await cl.Message(content=f"❓ **Commande inconnue:** `{command}`\n\nUtilisez `/help` pour voir les commandes disponibles.").send()

async def show_status():
    """Show detailed system status"""
    loop = asyncio.get_event_loop()
    status = await loop.run_in_executor(None, llm_interface.get_system_status)
    
    # Ollama status
    ollama_chat = "✅ Connecté" if status['ollama_chat'] else "❌ Déconnecté"
    ollama_embed = "✅ Connecté" if status['ollama_embeddings'] else "❌ Déconnecté"
    
    # Vector store status
    vector_stats = status.get('vector_store', {})
    
    # Documents status
    doc_details = []
    for doc_name, doc_info in status['documents'].items():
        company_name = config.get_company_name(doc_name)
        pdf_status = "✅" if doc_info['pdf_exists'] else "❌"
        json_status = "✅" if doc_info['json_exists'] else "❌"
        doc_details.append(f"   - **{company_name}**: PDF {pdf_status} | Traité {json_status}")
    
    status_message = f"""# 📊 Statut détaillé du système

## 🤖 Ollama:
- **Modèle de chat** ({config.models.chat_model}): {ollama_chat}
- **Modèle d'embeddings** ({config.models.embedding_model}): {ollama_embed}

## 🔍 Base vectorielle:
- **Collections**: {vector_stats.get('total_collections', 0)}
- **Documents indexés**: {vector_stats.get('total_documents', 0)}
- **Chunks totaux**: {vector_stats.get('total_chunks', 0)}

## 📚 Documents:
{chr(10).join(doc_details)}

## ⚙️ Configuration:
- **Taille des chunks**: {config.vector_store.chunk_size}
- **Chevauchement**: {config.vector_store.overlap}
- **Température LLM**: {config.models.temperature}
"""
    
    await cl.Message(content=status_message).send()

async def show_documents():
    """Show available documents"""
    loop = asyncio.get_event_loop()
    available_docs = await loop.run_in_executor(None, llm_interface.get_available_documents)
    
    if not available_docs:
        await cl.Message(content="❌ **Aucun document disponible**\n\nUtilisez `/process` pour traiter les documents.").send()
        return
    
    doc_list = []
    for doc_name, company_name in available_docs.items():
        # Get document config for more info
        try:
            doc_config = config.get_document_by_name(doc_name)
            doc_list.append(f"- **{company_name}** ({doc_config.content_type}): {doc_config.description}")
        except:
            doc_list.append(f"- **{company_name}**: {doc_name}")
    
    message = f"""# 📚 Documents disponibles

{chr(10).join(doc_list)}

💡 **Astuce:** Vous pouvez mentionner un restaurant spécifique dans votre question pour obtenir des informations ciblées, ou poser une question générale pour comparer les options.
"""
    
    await cl.Message(content=message).send()

async def process_documents():
    """Process/reprocess all documents"""
    processing_msg = cl.Message(content="🔄 **Traitement des documents en cours...**\n\nCela peut prendre quelques minutes.")
    await processing_msg.send()
    
    try:
        # Run processing in background
        success = await asyncio.get_event_loop().run_in_executor(None, pipeline.process_all_documents)
        
        # Remove processing message
        try:
            await processing_msg.remove()
        except:
            pass  # Ignore if already removed
        
        if success:
            # Update system status
            status = llm_interface.get_system_status()
            cl.user_session.set("system_status", status)
            
            status_msg = f"""✅ **Documents traités avec succès!**

📊 **Nouveau statut:**
- **Collections**: {status['vector_store'].get('total_collections', 0)}
- **Documents indexés**: {status['vector_store'].get('total_documents', 0)}

Le système est prêt à répondre à vos questions! 🎉"""
            
            await cl.Message(content=status_msg).send()
        else:
            await cl.Message(content="❌ **Erreur lors du traitement des documents**\n\nVérifiez les logs pour plus de détails.").send()
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        try:
            await processing_msg.remove()
        except:
            pass  # Ignore if already removed
        await cl.Message(content=f"❌ **Erreur**: {str(e)}").send()

async def show_help():
    """Show help message"""
    help_message = """# 🆘 Aide - Pizzeria RAG

## 💬 Questions normales:
Posez simplement votre question en français!
- "Quelles pizzas avez-vous?"
- "Prix de la Margherita?"
- "Options végétariennes chez Marco Fuso?"

## 🔧 Commandes spéciales:
- `/status` - Statut détaillé du système
- `/documents` - Liste des documents disponibles  
- `/process` - Traiter/retraiter les documents
- `/help` - Afficher cette aide

## 💡 Conseils:
- Mentionnez un restaurant spécifique pour des résultats ciblés
- Posez des questions générales pour comparer les options
- Soyez spécifique dans vos questions pour de meilleurs résultats

## 🔧 En cas de problème:
1. Vérifiez que Ollama fonctionne: `ollama serve`
2. Traitez les documents: `/process`
3. Vérifiez le statut: `/status`
"""
    
    await cl.Message(content=help_message).send()

if __name__ == "__main__":
    logger.info("🍕 Starting Chainlit Pizzeria RAG App")
