# 🍕 Pizzeria RAG - Assistant de Recherche Multi-Documents

<div align="center">
  <img src="assets/pizza_asset.webp" alt="Pizzeria RAG System" width="400">
</div>

Un système RAG (Retrieval-Augmented Generation) moderne pour interroger plusieurs menus de pizzerias avec Ollama et ChromaDB.

## 🚀 Fonctionnalités

- **RAG Multi-Documents**: Recherche simultanée dans plusieurs menus de pizzerias
- **Interface Moderne**: Chainlit et Gradio pour une expérience utilisateur optimale
- **Local-First**: Utilise Ollama pour les embeddings et LLM (pas de dépendance cloud)
- **Architecture Modulaire**: Code propre, maintenable et extensible
- **Processing Intelligent**: Extraction et indexation automatique des PDFs

## 🛠️ Architecture

```
pizzeria-RAG/
├── assets/                 # Images et ressources
│   └── pizza_asset.webp
├── config/                 # Configuration centralisée
│   └── config.py
├── data/                   # Données traitées et base vectorielle
│   ├── processed/         # JSONs structurés des documents
│   └── vector_db/         # Base vectorielle ChromaDB (git-ignoré)
├── docs/                   # Documentation et sources (git-ignoré)
│   └── raw_pdfs/          # Documents PDF sources
├── logs/                   # Fichiers de logs (git-ignoré)
├── processors/             # Traitement des documents
│   ├── __init__.py
│   └── document_processor.py
├── src/                    # Code source principal
│   ├── __init__.py
│   ├── apps/              # Applications utilisateur
│   │   ├── __init__.py
│   │   ├── chainlit_app.py    # Interface chat moderne
│   │   └── gradio_app.py      # Interface web alternative
│   ├── core/              # Fonctionnalités principales
│   │   ├── pipeline.py        # Pipeline principal
│   │   ├── rag_engine.py      # Moteur RAG complet
│   │   └── vector_store.py    # Gestion embeddings & ChromaDB
│   └── extractors/        # Extracteurs de contenu
│       ├── __init__.py
│       ├── base_extractor.py
│       ├── ocr_extractor.py
│       ├── pdf_extractor.py
│       ├── recipe_extractor.py
│       ├── table_extractor.py
│       └── text_extractor.py
├── chainlit.md             # Configuration Chainlit
├── requirements.txt        # Dépendances Python
├── start_chainlit.sh       # Script de lancement Chainlit
├── start_gradio.sh         # Script de lancement Gradio
├── .gitignore             # Fichiers ignorés par Git
└── README.md              # Documentation du projet
```

## 📋 Prérequis

1. **Ollama** installé et configuré
2. **Python 3.9+**
3. **Modèles Ollama**:
   - `llama3.2:latest` (chat)
   - `mxbai-embed-large` (embeddings)

## 🔧 Installation

1. **Cloner le projet**:
```bash
git clone <repository-url>
cd pizzeria-RAG
```

2. **Créer l'environnement virtuel**:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**:
```bash
pip install -r requirements.txt
```

4. **Démarrer Ollama** (dans un terminal séparé):
```bash
ollama serve
```

5. **Télécharger les modèles**:
```bash
ollama pull llama3.2:latest
ollama pull mxbai-embed-large
```

## 🎯 Utilisation

### Interface Chainlit (Recommandée)
```bash
./start_chainlit.sh
```
**Accès:** http://localhost:8000

**Fonctionnalités:**
- Chat interactif moderne
- Commandes système intégrées (`/status`, `/documents`, `/process`, `/help`)
- Affichage en temps réel du statut du système
- Interface épurée sans informations techniques parasites

### Interface Gradio (Alternative)
```bash
./start_gradio.sh
```
**Accès:** http://localhost:7860

**Fonctionnalités:**
- Interface web simple et intuitive
- Chat avec avatars utilisateur/assistant
- Onglet d'aide intégré
- Parfait pour les démos et tests rapides

### Utilisation directe en Python
```python
from src.core.rag_engine import LLMInterface

llm = LLMInterface()
result = llm.answer_question("Quelles pizzas végétariennes avez-vous?")
print(result['answer'])
```

## 💬 Exemples de Questions

- **Générales**: "Quelles pizzas avez-vous?"
- **Spécifiques**: "Chez Anchor Pizza, quel est le prix de la Margherita?"
- **Comparatives**: "Comparez les prix entre Marco Fuso et Anchor Pizza"
- **Détaillées**: "Quels sont les ingrédients du Triomphe végé?"

## 🔧 Commandes Système (Chainlit)

- `/status` - Statut détaillé du système
- `/documents` - Liste des documents disponibles
- `/process` - Traiter/retraiter les documents
- `/help` - Aide complète

## 📁 Données

Le système traite automatiquement les PDFs dans `docs/raw_pdfs/`:
- **Anchor Pizza**: Menu et services
- **Marco Fuso**: Recettes et techniques

Les données traitées sont stockées dans:
- `data/processed/` - JSONs structurés
- `data/vector_db/` - Base vectorielle ChromaDB

## ⚙️ Configuration

Modifiez `config/config.py` pour ajuster:
- Modèles Ollama utilisés
- Paramètres de chunking
- Température du LLM
- Ports et endpoints

### Sécurité et Git:
- **Dossiers protégés**: `data/`, `docs/`, `logs/` exclus du versioning
- **Base vectorielle**: ChromaDB stockée localement et non exposée
- **Fichiers sensibles**: `.env`, logs et caches automatiquement ignorés
- **Historique propre**: Suppression complète des fichiers sensibles du Git

## 🐛 Dépannage

### Problèmes courants:

1. **Ollama non connecté**:
   ```bash
   ollama serve
   curl http://localhost:11434/api/tags
   ```

2. **Modèles manquants**:
   ```bash
   ollama list
   ollama pull llama3.2:latest
   ollama pull mxbai-embed-large
   ```

3. **Scripts de lancement**:
   ```bash
   # Rendre les scripts exécutables
   chmod +x start_chainlit.sh start_gradio.sh
   
   # Vérifier l'environnement virtuel
   source .venv/bin/activate
   ```

4. **Documents non indexés**:
   - Utilisez `/process` dans Chainlit
   - Ou vérifiez `data/vector_db/`

5. **Problèmes de performance**:
   - Réduisez `chunk_size` dans la config
   - Vérifiez la RAM disponible

6. **Erreurs Gradio**:
   ```bash
   # Mise à jour Gradio si nécessaire
   pip install --upgrade gradio
   ```

## 🚀 Développement

### Améliorations récentes:
- **Interface Chainlit épurée**: Système de statut déplacé vers les commandes système
- **Réponses utilisateur propres**: Suppression des informations techniques des réponses
- **Architecture refactorisée**: Organisation claire du code en modules
- **Gestion des avatars**: Interfaces utilisateur modernes avec avatars
- **Scripts de lancement**: Démarrage simplifié avec vérifications automatiques

### Structure du code:
- **Modulaire**: Chaque composant a sa responsabilité
- **Type Hints**: Code typé pour une meilleure maintenance
- **Logging**: Traçabilité complète des opérations
- **Async/Await**: Interface non-bloquante pour une meilleure UX
- **Gestion d'erreurs**: Robustesse et récupération automatique

### Ajouter de nouveaux documents:
1. Placer le PDF dans `docs/raw_pdfs/`
2. Ajouter la configuration dans `config/config.py`
3. Relancer le traitement via `/process` (Chainlit) ou l'interface Gradio

## 📄 Licence

Ce projet est sous licence MIT. Voir `LICENSE` pour plus de détails.

## 🤝 Contribution

Les contributions sont bienvenues! Merci de:
1. Fork le projet
2. Créer une branche feature
3. Commit vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📞 Support

En cas de problème:
1. Vérifiez la section Dépannage
2. Consultez les logs de l'application
3. Ouvrez une issue sur GitHub

---

**Développé avec ❤️ pour une expérience RAG moderne et efficace**
