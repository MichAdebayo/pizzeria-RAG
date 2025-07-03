#!/bin/bash
# Launch script for Pizzeria RAG Chainlit interface

echo "🍕 Démarrage de l'interface Pizzeria RAG avec Chainlit..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama ne semble pas fonctionner. Assurez-vous qu'il est démarré avec 'ollama serve'"
    echo "Tentative de démarrage quand même..."
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Environnement virtuel non activé. Activation automatique..."
    source .venv/bin/activate 2>/dev/null || echo "❌ Impossible d'activer l'environnement virtuel"
fi

# Start Chainlit
echo "🚀 Lancement de Chainlit..."
chainlit run src/apps/chainlit_app.py --host 0.0.0.0 --port 8000

# If that fails, try with watch mode for development
if [ $? -ne 0 ]; then
    echo "🔄 Tentative en mode développement..."
    chainlit run chainlit_app.py --watch
fi
