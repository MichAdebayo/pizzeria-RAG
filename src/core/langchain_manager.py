from typing import Dict, List, Any, Optional
import yaml
import logging
from pathlib import Path

try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_community.llms import Ollama
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain imports not available: {e}")
    LANGCHAIN_AVAILABLE = False

from src.core.vector_store_manager import VectorStoreManager

class LangChainManager:
    """Central LangChain orchestrator for the pizzeria RAG system"""
    
    def __init__(self, agents_config_path: str):
        with open(agents_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embeddings = self._setup_embeddings()
        self.llm = self._setup_llm()
        self.vector_store_manager = VectorStoreManager(self.config, self.embeddings)
        self.memory = self._setup_memory()
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        self.agent_executor = self._setup_executor()
    
    def _setup_embeddings(self):
        """Setup embedding model"""
        embedding_config = self.config['langchain_config']['embeddings']
        
        if embedding_config['provider'] == 'huggingface':
            return HuggingFaceEmbeddings(
                model_name=embedding_config['model']
            )
        elif embedding_config['provider'] == 'ollama':
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(
                model=embedding_config['model']
            )
    
    def _setup_llm(self):
        """Setup LLM"""
        llm_config = self.config['langchain_config']['llm']
        
        if llm_config['provider'] == 'ollama':
            return Ollama(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                num_predict=llm_config['max_tokens']
            )
    
    def _setup_memory(self):
        """Setup conversation memory"""
        memory_config = self.config.get('memory', {})
        return ConversationBufferWindowMemory(
            k=memory_config.get('k', 10),
            return_messages=memory_config.get('return_messages', True),
            memory_key="chat_history"
        )
    
    def _setup_tools(self):
        """Initialize all LangChain tools"""
        tools = []
        try:
            # Import tools dynamically to avoid import errors
            try:
                from src.tools import PizzaSearchTool, AllergenCheckTool
                
                # Try to import the v2 tools, fallback to original if needed
                try:
                    from src.tools.ingredient_lookup_tool_v2 import IngredientLookupTool
                    from src.tools.nutrition_info_tool_v2 import NutritionInfoTool
                except ImportError:
                    from src.tools import IngredientLookupTool, NutritionInfoTool
                
                tools_available = True
                self.logger.info("Successfully imported all tools")
            except ImportError as e:
                logging.warning(f"Tools not available, using fallback: {e}")
                tools_available = False
            
            if tools_available:
                vector_stores = self.vector_store_manager.get_all_stores()
                
                # Debug logging
                self.logger.info(f"Available vector stores: {list(vector_stores.keys())}")
                self.logger.info(f"Tools config: {list(self.config.get('tools', {}).keys())}")
                
                tool_classes = {
                    'pizza_search': PizzaSearchTool,
                    'allergen_check': AllergenCheckTool,
                    'ingredient_lookup': IngredientLookupTool,
                    'nutrition_info': NutritionInfoTool
                }
                
                for tool_name, tool_config in self.config.get('tools', {}).items():
                    if tool_name in tool_classes:
                        vector_store_name = tool_config.get('vector_store', 'pizza_descriptions')
                        
                        self.logger.info(f"Tool {tool_name} looking for vector store: {vector_store_name}")
                        self.logger.info(f"Vector stores type: {type(vector_stores)}")
                        self.logger.info(f"Vector stores content: {vector_stores}")
                        
                        vector_store = vector_stores.get(vector_store_name)
                        self.logger.info(f"Retrieved vector store: {vector_store} (type: {type(vector_store)})")
                        
                        if vector_store is not None:
                            try:
                                tool_instance = tool_classes[tool_name](
                                    config=tool_config,
                                    vector_store=vector_store
                                )
                                tools.append(tool_instance)
                                self.logger.info(f"Successfully initialized tool: {tool_name}")
                            except Exception as e:
                                self.logger.error(f"Failed to create tool {tool_name}: {e}")
                        else:
                            self.logger.warning(f"Vector store {vector_store_name} not found for tool {tool_name}")
                            self.logger.warning(f"Available stores: {list(vector_stores.keys())}")
            else:
                # Create fallback tools
                tools = [MockTool()]
        
        except Exception as e:
            self.logger.error(f"Failed to setup tools: {e}")
            # Create fallback tools
            tools = [MockTool()]
        
        return tools
    
    def _setup_agent(self):
        """Create ReAct agent with custom prompt"""
        prompt = PromptTemplate.from_template("""
        Vous êtes un assistant pour les pizzerias. Vous aidez les clients avec:
        - Les informations sur les pizzas, leurs sources (pizzerias) et les prix
        - Les listes d'ingrédients et informations allergènes (CRITIQUE - toujours vérifier)
        - Les informations nutritionnelles
        - Questions générales sur les pizzas

        RÈGLES DE SÉCURITÉ:
        - Pour les questions allergènes, TOUJOURS utiliser l'outil allergen_safety_check
        - Si la confiance est < 99%, demander au client de contacter directement le restaurant
        - Être précis et utile avec les informations d'ingrédients
        - Toujours citer vos sources ET la pizzeria d'origine
        - IMPORTANT: Indiquez clairement quelle pizzeria propose chaque pizza

        Vous avez accès aux outils suivants:
        {tools}

        Utilisez EXACTEMENT ce format:

        Question: la question d'entrée à laquelle vous devez répondre
        Pensée: vous devez toujours réfléchir à ce qu'il faut faire
        Action: l'action à entreprendre, doit être l'un de [{tool_names}]
        Entrée d'action: l'entrée de l'action
        Observation: le résultat de l'action
        ... (cette Pensée/Action/Entrée d'action/Observation peut se répéter N fois)
        Pensée: Je connais maintenant la réponse finale
        Réponse finale: la réponse finale à la question d'entrée originale

        Commencez!

        Question: {input}
        Pensée: {agent_scratchpad}
        """)
        
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def _setup_executor(self):
        """Setup agent executor"""
        executor_config = self.config['langchain_config']['agent_executor']
        
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            max_iterations=executor_config['max_iterations'],
            early_stopping_method=executor_config['early_stopping_method'],
            verbose=executor_config['verbose'],
            handle_parsing_errors=True
        )
    
    def query(self, question: str) -> dict:
        """Main query interface"""
        try:
            response = self.agent_executor.invoke({"input": question})
            return {
                'status': 'success',
                'response': response['output'],
                'chat_history': self.memory.chat_memory.messages
            }
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                'status': 'error',
                'response': f"Désolé, une erreur s'est produite: {str(e)}",
                'error': str(e)
            }


# Mock classes for fallback when components are not available
class MockTool:
    def __init__(self):
        self.name = "mock_tool"
        self.description = "Tool de développement"
    
    def run(self, query):
        return f"Recherche pour: {query} (Mode développement - LangChain non configuré)"
    
    def get(self, key, default=None):
        """Mock the get method for compatibility"""
        return getattr(self, key, default)


class MockAgent:
    def __call__(self, inputs):
        return {"output": "Réponse de l'agent de développement"}


class MockExecutor:
    def invoke(self, inputs):
        return {"output": f"Réponse de développement pour: {inputs.get('input', '')}"}


class MockMemory:
    def __init__(self):
        self.chat_memory = []


class MockVectorStoreManager:
    def get_all_stores(self):
        return {}