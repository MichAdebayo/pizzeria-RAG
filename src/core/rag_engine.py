"""
Modular LLM interface for the pizzeria RAG system
Based on the working simple_llm_interface.py, now supporting multiple documents
"""

import logging
from typing import Dict, List, Optional, Union
import ollama
from config.config import config
from src.core.vector_store import VectorStore

class LLMInterface:
    """LLM interface for pizzeria RAG using Ollama, supporting multiple documents"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chat_model = config.models.chat_model
        self.embedding_model = config.models.embedding_model
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # Initialize vector store with Ollama embeddings
        self.vector_store = VectorStore()
        
    def _test_ollama_connection(self):
        """Test if Ollama is running and models are available (same as working simple version)"""
        try:
            # Test chat model
            ollama.chat(model=self.chat_model, messages=[{"role": "user", "content": "test"}])
            self.logger.info(f"✅ Ollama chat model '{self.chat_model}' is ready")
            
            # Test embedding model
            ollama.embeddings(model=self.embedding_model, prompt="test")
            self.logger.info(f"✅ Ollama embedding model '{self.embedding_model}' is ready")
            
        except Exception as e:
            self.logger.error(f"❌ Ollama connection failed: {e}")
            self.logger.info("Make sure Ollama is running: ollama serve")
            self.logger.info(f"And that models are available: ollama pull {self.chat_model} && ollama pull {self.embedding_model}")
    
    def get_context(self, query: str, document_names: Optional[Union[str, List[str]]] = None, 
                    max_chunks: int = 5) -> Dict:
        """Get relevant context from vector store with company grouping and diversity"""
        # If no specific documents requested, search more results to ensure diversity
        search_limit = max_chunks * 2 if document_names is None else max_chunks
        
        search_results = self.vector_store.search(query, document_names=document_names, n_results=search_limit)
        
        # Group context by document/company with diversity control
        context_by_company = {}
        documents_used = []
        company_counts = {}
        max_per_company = max(2, max_chunks // 2) if document_names is None else max_chunks
        
        for result in search_results.get('results', []):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            doc_name = result.get('document_name', 'unknown')
            
            if doc_name not in documents_used:
                documents_used.append(doc_name)
            
            # Get clean company name using config method
            company_name = config.get_company_name(doc_name)
            
            # Limit results per company to ensure diversity
            if company_name not in company_counts:
                company_counts[company_name] = 0
            
            if company_counts[company_name] >= max_per_company:
                continue
                
            if company_name not in context_by_company:
                context_by_company[company_name] = []
            
            context_part = f"[Page {metadata.get('page', 'N/A')}] {content}"
            context_by_company[company_name].append(context_part)
            company_counts[company_name] += 1
            
            # Stop if we have enough total results
            total_results = sum(len(contexts) for contexts in context_by_company.values())
            if total_results >= max_chunks:
                break
        
        return {
            'context_by_company': context_by_company,
            'documents_used': documents_used,
            'has_multiple_companies': len(context_by_company) > 1
        }
    
    def _detect_company_in_query(self, query: str) -> Optional[str]:
        """Detect if a specific company is mentioned in the query - be very conservative"""
        query_lower = query.lower()
        
        # Only detect companies if they are explicitly mentioned with clear indicators
        for doc in config.documents:
            # Get the clean company name (e.g., "Anchor Pizza", "Marco Fuso")
            company_name = config.get_company_name(doc.name).lower()
            
            # Check for explicit company mentions with context
            # Look for patterns like "chez [company]", "[company] a-t-il", etc.
            explicit_patterns = [
                f"chez {company_name}",
                f"à {company_name}",
                f"{company_name} a-t-il",
                f"{company_name} avez-vous",
                f"{company_name} propose",
                f"restaurant {company_name}",
                f"pizzeria {company_name}"
            ]
            
            for pattern in explicit_patterns:
                if pattern in query_lower:
                    return doc.name
            
            # Also check for individual company name parts, but only if very specific
            company_parts = company_name.split()
            for part in company_parts:
                if len(part) > 4:  # Only check longer words to avoid false positives
                    # Must be followed by clear restaurant/company indicators
                    company_indicators = [
                        f"{part} pizza",
                        f"{part} restaurant", 
                        f"chez {part}",
                        f"pizzeria {part}"
                    ]
                    for indicator in company_indicators:
                        if indicator in query_lower:
                            return doc.name
        
        return None
    
    def create_prompt(self, user_question: str, context_data: Dict, document_names: Optional[Union[str, List[str]]] = None) -> str:
        """Create a prompt with context for the LLM (enhanced for multi-company and allergens)"""
        
        context_by_company = context_data['context_by_company']
        has_multiple_companies = context_data['has_multiple_companies']
        
        # Get allergen information for all contexts
        allergen_info = self.get_allergen_info_for_context(context_data)
        
        # Format context based on whether we have multiple companies
        if has_multiple_companies:
            # Multi-company format
            context_text = "INFORMATIONS DE NOS DIFFÉRENTS SERVICES:\n\n"
            for company_name, company_contexts in context_by_company.items():
                context_text += f"🍕 {company_name.upper()}:\n"
                for ctx in company_contexts:
                    context_text += f"   {ctx}\n"
                
                # Add allergen information for this company
                if company_name in allergen_info and allergen_info[company_name]:
                    allergens = ", ".join(allergen_info[company_name])
                    context_text += f"   ⚠️ ALLERGÈNES DÉTECTÉS: {allergens}\n"
                else:
                    context_text += f"   ✅ AUCUN ALLERGÈNE MAJEUR DÉTECTÉ\n"
                context_text += "\n"
        else:
            # Single company format (like the original working version)
            company_name = list(context_by_company.keys())[0] if context_by_company else "Restaurant"
            context_text = f"INFORMATIONS DE {company_name.upper()}:\n\n"
            for company_contexts in context_by_company.values():
                for ctx in company_contexts:
                    context_text += f"{ctx}\n\n"
            
            # Add allergen information for single company
            if company_name in allergen_info and allergen_info[company_name]:
                allergens = ", ".join(allergen_info[company_name])
                context_text += f"⚠️ ALLERGÈNES DÉTECTÉS: {allergens}\n\n"
            else:
                context_text += f"✅ AUCUN ALLERGÈNE MAJEUR DÉTECTÉ\n\n"
        
        # Add comprehensive allergen list for reference
        allergen_list = config.allergen.allergens_list or []
        all_allergens = ", ".join(allergen_list)
        context_text += f"\nLISTE COMPLÈTE DES ALLERGÈNES À SURVEILLER:\n{all_allergens}\n\n"
        
        # Extract user allergens from question for targeted warnings
        user_allergens = self.extract_user_allergens_from_question(user_question)
        if user_allergens:
            context_text += f"⚠️ ATTENTION: Le client a mentionné être allergique à: {', '.join(user_allergens)}\n\n"
        
        # Determine response strategy
        if has_multiple_companies:
            system_role = "Tu es un assistant du groupe de pizzerias. Nous avons plusieurs restaurants avec des menus différents."
            instructions = """INSTRUCTIONS:
- Nous sommes un groupe de pizzerias avec plusieurs restaurants
- Si la question est générale, présente les options de chaque restaurant séparément
- Si un restaurant spécifique est mentionné, concentre-toi sur celui-ci
- Sois précis sur quel restaurant offre quoi
- Format: "Chez [Nom Restaurant]: [info]" pour chaque restaurant
- PRIORITÉ ABSOLUE: TOUJOURS inclure les informations d'allergènes pour chaque pizza mentionnée
- Si des allergènes sont détectés, AVERTIS CLAIREMENT le client avec des emojis ⚠️
- Si le client mentionne des allergies spécifiques, VÉRIFIE LA COMPATIBILITÉ
- Suggère des alternatives sans allergènes si nécessaire
- Utilise le format: "⚠️ Allergènes: [liste]" ou "✅ Aucun allergène majeur détecté"
- Reste dans le rôle d'un assistant de groupe de pizzerias"""
        else:
            system_role = "Tu es un assistant de pizzeria. Réponds en français, sois précis et utile."
            instructions = """INSTRUCTIONS:
- Réponds uniquement en français
- Base-toi uniquement sur les informations du contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- PRIORITÉ ABSOLUE: TOUJOURS inclure les informations d'allergènes pour chaque pizza mentionnée
- Si des allergènes sont détectés, AVERTIS CLAIREMENT le client avec des emojis ⚠️
- Si le client mentionne des allergies spécifiques, VÉRIFIE LA COMPATIBILITÉ
- Utilise le format: "⚠️ Allergènes: [liste]" ou "✅ Aucun allergène majeur détecté"
- Suggère des alternatives sans allergènes si nécessaire
- Sois précis et utile
- Reste dans le rôle d'un assistant de pizzeria"""
        
        prompt = f"""{system_role}

{context_text}

QUESTION DU CLIENT: {user_question}

{instructions}

RÉPONSE:"""
        
        return prompt
    
    def query_llm(self, prompt: str) -> str:
        """Query Ollama with the prompt (same as working simple version)"""
        try:
            response = ollama.chat(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Tu es un assistant de pizzeria. Réponds en français, sois précis et utile."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                options={
                    "temperature": config.models.temperature,
                    "top_p": config.models.top_p,
                    "num_predict": config.models.num_predict
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when Ollama is not available"""
        # Extract question from prompt
        question = "votre question"
        if "QUESTION DU CLIENT:" in prompt:
            question = prompt.split('QUESTION DU CLIENT: ')[1].split('INSTRUCTIONS:')[0].strip()
        
        context_data = self.get_context(question, max_chunks=2)
        
        return f"""🍕 Assistant Pizzeria (Mode Hors Ligne)

Désolé, je ne peux pas accéder au modèle de langage actuellement.

Voici ce que j'ai trouvé dans nos documents pour "{question}":

{str(context_data['context_by_company']) if context_data['context_by_company'] else "Aucune information pertinente trouvée dans notre base de données."}

Pour activer la fonctionnalité complète:
1. Lancez Ollama: ollama serve
2. Vérifiez que les modèles sont disponibles: ollama list"""
    
    def answer_question(self, question: str, document_names: Optional[Union[str, List[str]]] = None, user_allergens: Optional[List[str]] = None) -> Dict:
        """Complete RAG pipeline: retrieve context and generate answer (enhanced for multi-company and allergens)"""
        try:
            # Auto-detect user allergens from the question if not provided
            if user_allergens is None:
                user_allergens = self.extract_user_allergens_from_question(question)
            
            # Auto-detect company if not specified and question contains company name
            if document_names is None:
                detected_company = self._detect_company_in_query(question)
                if detected_company:
                    document_names = detected_company
            
            # Step 1: Get relevant context with company grouping
            context_data = self.get_context(question, document_names=document_names)
            
            # Step 2: Get allergen information
            allergen_info = self.get_allergen_info_for_context(context_data)
            
            # Step 3: Create prompt with allergen awareness
            prompt = self.create_prompt(question, context_data, document_names=document_names)
            
            # Step 4: Query LLM
            answer = self.query_llm(prompt)
            
            # Step 5: Add allergen analysis to the response
            allergen_analysis = ""
            if user_allergens:
                allergen_analysis = self.suggest_alternatives_for_allergens(user_allergens, context_data)
                self.logger.info(f"🚨 User allergens detected: {', '.join(user_allergens)}")
            
            # Always add detected allergens summary for transparency
            if allergen_info:
                allergen_summary = "\n\n🧾 **Résumé allergènes détectés:**\n"
                for company_name, allergens in allergen_info.items():
                    if allergens:
                        allergen_summary += f"• {company_name}: {', '.join(allergens)}\n"
                    else:
                        allergen_summary += f"• {company_name}: Aucun allergène majeur détecté\n"
                answer += allergen_summary
            
            # Add user-specific allergen analysis if provided
            if allergen_analysis:
                answer += allergen_analysis
            
            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "context_used": context_data['context_by_company'],
                "has_context": bool(context_data['context_by_company']),
                "searched_documents": document_names or [doc.name for doc in config.documents],
                "has_multiple_companies": context_data['has_multiple_companies'],
                "companies_found": list(context_data['context_by_company'].keys()),
                "allergen_info": allergen_info,
                "user_allergens": user_allergens or []
            }
            
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {e}")
            return {
                "status": "error",
                "question": question,
                "answer": f"Désolé, une erreur s'est produite: {str(e)}",
                "context_used": {},
                "has_context": False,
                "searched_documents": [],
                "has_multiple_companies": False,
                "companies_found": [],
                "allergen_info": {},
                "user_allergens": user_allergens or []
            }
    
    def extract_user_allergens_from_question(self, question: str) -> List[str]:
        """Extract user allergens mentioned in the question"""
        question_lower = question.lower()
        user_allergens = []
        
        # Check for allergen mentions in the question
        allergen_keywords = config.allergen.get_allergen_keywords()
        
        # First, look for general allergen context indicators
        allergen_context_indicators = [
            "allergique", "allergie", "allergies", "allergène", "allergènes",
            "intolerance", "intolérant", "sans", "éviter", "peut pas manger",
            "ne peux pas manger", "ne mange pas"
        ]
        
        has_allergen_context = any(indicator in question_lower for indicator in allergen_context_indicators)
        
        # If there's allergen context, be more liberal in detecting allergens
        if has_allergen_context:
            for allergen, keywords in allergen_keywords.items():
                for keyword in keywords:
                    if keyword in question_lower:
                        # Check for negative context (allergic to, can't eat, etc.)
                        allergen_patterns = [
                            f"allergique au {keyword}",
                            f"allergique à {keyword}",
                            f"allergique aux {keyword}",
                            f"allergie au {keyword}",
                            f"allergie à {keyword}",
                            f"allergie aux {keyword}",
                            f"pas de {keyword}",
                            f"sans {keyword}",
                            f"éviter {keyword}",
                            f"ne peut pas manger {keyword}",
                            f"ne peux pas manger {keyword}",
                            f"ne mange pas {keyword}",
                            f"intolérant au {keyword}",
                            f"intolérant à {keyword}",
                            f"intolerance au {keyword}",
                            f"intolerance à {keyword}",
                            f"éviter les {keyword}",
                            f"allergie aux {keyword}",
                            f"peut pas manger de {keyword}",
                            f"ne peux pas manger de {keyword}"
                        ]
                        
                        # Check specific patterns first
                        pattern_found = False
                        for pattern in allergen_patterns:
                            if pattern in question_lower:
                                if allergen not in user_allergens:
                                    user_allergens.append(allergen)
                                pattern_found = True
                                break
                        
                        # If no specific pattern found but allergen context exists
                        # and keyword is mentioned, consider it an allergen
                        if not pattern_found and has_allergen_context:
                            # Additional check for compound mentions like "allergique au gluten et au lait"
                            if any(context in question_lower for context in ["allergique", "allergie", "sans", "éviter"]):
                                if allergen not in user_allergens:
                                    user_allergens.append(allergen)
        
        return user_allergens
    
    def is_allergen_related_question(self, question: str) -> bool:
        """Check if the question is related to allergens"""
        allergen_indicators = [
            "allergique", "allergie", "allergies", "allergène", "allergènes",
            "intolerance", "intolérant", "sans", "éviter", "peut pas manger",
            "gluten", "lactose", "végétalien", "vegan", "végétarien"
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in allergen_indicators)

    def get_available_documents(self) -> Dict[str, str]:
        """Get available documents for selection"""
        return config.get_available_documents()
    
    def get_system_status(self) -> Dict:
        """Get system status information"""
        status = {
            "ollama_chat": False,
            "ollama_embeddings": False,
            "vector_store": {},
            "documents": {}
        }
        
        try:
            # Test chat model
            ollama.chat(model=self.chat_model, messages=[{"role": "user", "content": "test"}])
            status["ollama_chat"] = True
        except:
            status["ollama_chat"] = False
        
        try:
            # Test embedding model
            ollama.embeddings(model=self.embedding_model, prompt="test")
            status["ollama_embeddings"] = True
        except:
            status["ollama_embeddings"] = False
        
        # Vector store stats
        try:
            status["vector_store"] = self.vector_store.get_stats()
        except Exception as e:
            status["vector_store"] = {"error": str(e)}
        
        # Document status
        for doc_config in config.documents:
            from pathlib import Path
            status["documents"][doc_config.name] = {
                "pdf_exists": Path(doc_config.pdf_path).exists(),
                "json_exists": Path(doc_config.processed_json_path).exists(),
                "description": doc_config.description
            }
        
        return status
    
    def detect_allergens_in_text(self, text: str) -> List[str]:
        """Detect allergens present in ingredient text"""
        if not text:
            return []
        
        text_lower = text.lower()
        detected_allergens = []
        allergen_keywords = config.allergen.get_allergen_keywords()
        
        for allergen, keywords in allergen_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if allergen not in detected_allergens:
                        detected_allergens.append(allergen)
                    break  # Found this allergen, no need to check other keywords
        
        return detected_allergens
    
    def get_allergen_info_for_context(self, context_data: Dict) -> Dict[str, List[str]]:
        """Extract allergen information from context data"""
        allergen_info = {}
        
        for company_name, company_contexts in context_data['context_by_company'].items():
            company_allergens = set()
            
            for context in company_contexts:
                # Detect allergens in each context piece
                detected = self.detect_allergens_in_text(context)
                company_allergens.update(detected)
            
            allergen_info[company_name] = sorted(list(company_allergens))
        
        return allergen_info
    
    def suggest_alternatives_for_allergens(self, user_allergens: List[str], context_data: Dict) -> str:
        """Suggest pizza alternatives based on user's allergen restrictions"""
        if not user_allergens:
            return ""
        
        allergen_info = self.get_allergen_info_for_context(context_data)
        suggestions = []
        
        for company_name, company_allergens in allergen_info.items():
            # Check if any of the user's allergens are present
            has_allergens = any(allergen in company_allergens for allergen in user_allergens)
            
            if not has_allergens:
                suggestions.append(f"✅ {company_name}: Semble compatible avec vos restrictions")
            else:
                conflicting = [allergen for allergen in user_allergens if allergen in company_allergens]
                suggestions.append(f"⚠️ {company_name}: Contient {', '.join(conflicting)}")
        
        if suggestions:
            return "\n\n🔍 **Analyse allergènes:**\n" + "\n".join(suggestions)
        return ""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create LLM interface
    llm = LLMInterface()
    
    # Test questions (same as working simple version)
    test_questions = [
        "Quelles pizzas avez-vous au menu?",
        "Quel est le prix de la pizza Margherita?",
        "Avez-vous des pizzas végétariennes?"
    ]
    
    print("🔸 Testing with ALL documents:")
    for question in test_questions:
        print(f"\n🔸 Question: {question}")
        result = llm.answer_question(question)
        print(f"✅ Réponse: {result['answer']}")
        print(f"📄 Contexte utilisé: {'Oui' if result['has_context'] else 'Non'}")
        print(f"📚 Documents consultés: {result['searched_documents']}")
    
    print("\n" + "="*50)
    print("🔸 Testing with SPECIFIC document (marco_fuso):")
    for question in test_questions:
        print(f"\n🔸 Question: {question}")
        result = llm.answer_question(question, document_names="marco_fuso")
        print(f"✅ Réponse: {result['answer']}")
        print(f"📄 Contexte utilisé: {'Oui' if result['has_context'] else 'Non'}")
        print(f"📚 Documents consultés: {result['searched_documents']}")
