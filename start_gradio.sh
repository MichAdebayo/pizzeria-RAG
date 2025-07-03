#!/bin/bash
# Launch script for Pizzeria RAG Gradio interface

echo "🍕 Démarrage de l'interface Pizzeria RAG avec Gradio..."

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

# Start Gradio
echo "🚀 Lancement de Gradio..."
python src/apps/gradio_app.py

echo "✅ Application fermée"
