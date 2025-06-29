"""
LLM integration service for OrgGPT.

This module handles integration with language models via LangChain-Groq
and provides query processing, intent classification, and response generation.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from ..models.schemas import IntentType, QueryMetadata, ChatMessage
from ..core.config import settings, RAG_INTENTS, GENERAL_INTENTS, HYBRID_INTENTS

import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM operations and query processing."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.1
        )
        
        # Intent classification prompts
        self.intent_labels = [intent.value for intent in IntentType]
        
    def rewrite_query(self, query: str, chat_history: List[ChatMessage]) -> str:
        """
        Rewrite user query for better understanding and retrieval.
        
        Args:
            query: Original user query
            chat_history: Recent chat history for context
            
        Returns:
            str: Rewritten query
        """
        try:
            # Create context from recent chat history
            context = ""
            if chat_history:
                recent_messages = chat_history[-3:]  # Last 3 messages
                for msg in recent_messages:
                    context += f"{msg.role}: {msg.content}\n"
            
            system_prompt = QUERY_REWRITING_PROMPT = """
You are an expert query rewriting assistant for a document retrieval system. Your task is to transform user queries into optimal search queries that will retrieve the most relevant information from uploaded documents.

CORE PRINCIPLES:
1. Preserve the original intent and meaning completely
2. Optimize for document retrieval and semantic search
3. Use specific, searchable terminology
4. Incorporate relevant context from conversation history
5. Return only the rewritten query, nothing else

REWRITING STRATEGIES:

**For Vague/Ambiguous Queries:**
- Add context from chat history to clarify pronouns (it, this, that, these)
- Expand abbreviations and acronyms 
- Convert colloquial language to formal/technical terms
- Specify the domain or document type if unclear

**For Follow-up Questions:**
- Include the topic from previous conversation
- Replace pronouns with specific nouns from context
- Maintain conversation thread continuity

**For Incomplete Queries:**
- Add missing context that's implied from conversation
- Expand fragments into complete search terms
- Include relevant synonyms for better matching

**For Conversational Queries:**
- Remove conversational fillers ("can you", "please", "I want to know")
- Extract the core information need
- Convert questions to keyword-rich statements when beneficial

**For Technical Queries:**
- Preserve technical terminology exactly
- Add related technical terms that might appear in documents
- Include both formal and informal versions of technical concepts

**For Temporal Queries:**
- Add specific time periods if mentioned in context
- Include date-related keywords for time-sensitive information

**For Comparative Queries:**
- Ensure all comparison elements are clearly specified
- Add context for what's being compared

EXAMPLES:

Input: "What's the policy?"
Context: Previous message mentioned "remote work guidelines"
Output: "remote work policy guidelines requirements"

Input: "How do I do this?"
Context: Previous message about "API authentication setup"
Output: "API authentication setup configuration steps process"

Input: "Tell me about it"
Context: Previous message mentioned "quarterly sales report"
Output: "quarterly sales report analysis data metrics"

Input: "Can you explain the process?"
Context: Discussion about "employee onboarding"
Output: "employee onboarding process steps procedures workflow"

Input: "What are the requirements for this?"
Context: Previous message about "software deployment"
Output: "software deployment requirements specifications prerequisites"

Input: "Show me the latest version"
Context: Discussion about "API documentation"
Output: "latest API documentation version updates changes"

Input: "I need help with the error"
Context: Previous message mentioned "database connection timeout"
Output: "database connection timeout error troubleshooting fix"

Input: "What's new in v2?"
Context: Discussion about "mobile app features"
Output: "mobile app version 2 new features updates changes"

Input: "Compare these two options"
Context: Previous messages about "cloud storage solutions"
Output: "cloud storage solutions comparison features pricing options"

Input: "When was this implemented?"
Context: Discussion about "two-factor authentication"
Output: "two-factor authentication implementation date timeline"

SPECIAL CASES:

**Already Optimal Queries:** If the query is already specific, clear, and well-structured for search, return it unchanged.

**Multi-part Queries:** Break down complex queries into their core searchable components while maintaining relationships.

**Domain-Specific Queries:** Preserve industry jargon and technical terms that are likely to appear in documents.

**Contextual References:** Always resolve references like "this document", "that section", "the previous report" using chat history.

**Spelling/Grammar:** Correct obvious errors but preserve intentional technical terminology.

INSTRUCTIONS:
- Use the chat history to resolve ambiguous references
- Focus on nouns, verbs, and key concepts that would appear in documents  
- Remove conversational elements that don't aid in search
- Ensure the rewritten query would match relevant document content
- Keep queries concise but comprehensive
- If no rewriting is needed, return the original query

Return only the rewritten query without explanations, quotes, or additional formatting.
"""
            
            user_prompt = f"""Chat History:
{context}

Current Query: {query}

Rewritten Query:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            rewritten = response.content.strip()
            
            # Fallback to original query if rewriting fails
            if not rewritten or len(rewritten) < 3:
                return query
                
            return rewritten
            
        except Exception as e:
            print(f"Error rewriting query: {str(e)}")
            return query
    
    def classify_intent(
            self, 
            query: str,
            retrieved_docs: List[str],
            avg_relevance: float

        ) -> Tuple[IntentType, float]:
        """
        Classify the intent of a user query.
        
        Args:
            query: User query to classify
            
        Returns:
            Tuple[IntentType, float]: Classified intent and confidence score
        """
        try:
            # Create intent classification prompt
            system_prompt = """You are an intent classification system for a RAG (Retrieval-Augmented Generation) chatbot. Your task is to analyze a user query and retrieved documents to determine the most appropriate intent and response strategy.

## CLASSIFICATION CATEGORIES:

### RAG-REQUIRED INTENTS
Use when the query requires retrieving specific information from uploaded documents:
- document_search: Finding specific documents or files in the uploaded collection
- content_lookup: Looking up specific information within uploaded documents  
- document_summary: Summarizing uploaded documents or sections
- cross_reference: Finding information across multiple uploaded documents
- information_extraction: Extracting specific data points from uploaded documents
- fact_verification: Verifying claims against uploaded document content
- section_retrieval: Getting specific sections from uploaded documents
- keyword_search: Searching for terms/phrases in uploaded documents
- comparative_analysis: Comparing information between uploaded documents
- trend_analysis: Identifying patterns in uploaded document data
- evidence_finding: Finding supporting evidence from uploaded sources
- citation_request: Getting sources/references from uploaded documents
- quote_extraction: Extracting specific quotes or passages from documents
- data_retrieval: Retrieving specific data points or statistics from documents
- source_attribution: Identifying which document contains specific information

### GENERAL-KNOWLEDGE INTENTS
Use when the query can be answered using general knowledge (no document retrieval needed):
- greeting: Greetings, hellos, how are you
- chitchat: Casual conversation, small talk, personal questions
- farewell: Goodbyes, see you later, closing conversations
- gratitude: Thank you, appreciation, acknowledgments
- definitions: General definitions or explanations of concepts
- explanations: General explanations not requiring specific documents
- math_calculations: Mathematical problems or calculations
- science_facts: General science knowledge and facts
- history_general: General historical information
- geography: Information about places, countries, cities
- technology_general: General technology concepts and trends
- programming_help: General coding questions and programming concepts
- language_questions: Grammar, translation, language learning
- creative_writing: Story writing, poetry, creative content generation
- brainstorming: Idea generation, creative thinking exercises
- general_advice: Life advice, general recommendations
- trivia: Fun facts, quiz questions, general knowledge trivia
- current_events: General news, current affairs (from training data)
- cooking_recipes: General cooking and recipe information
- health_general: General health and wellness information
- entertainment: Movies, books, music, games recommendations
- travel_general: General travel advice and destination information
- hobby_interests: General hobby and interest discussions
- philosophy: Philosophical questions and discussions
- ethics: Ethical discussions and moral questions

### HYBRID INTENTS
Use when the query requires both uploaded document content AND general knowledge:
- document_explanation: Explaining concepts found in documents using general knowledge
- context_enhancement: Adding general context to document-specific information
- comparison_external: Comparing document content with general knowledge
- document_validation: Validating document claims against general knowledge
- knowledge_synthesis: Combining document facts with general understanding
- educational_support: Using documents for learning with general explanations
- research_assistance: Helping with research using both sources and general knowledge
- analysis_enhancement: Enhancing document analysis with broader knowledge
- recommendation_informed: Making recommendations based on both document data and general principles
## ANALYSIS CRITERIA:
1. Document Relevance Score (0-10): How relevant are the retrieved documents to the query?
2. Document Completeness Score (0-10): How completely do the documents answer the query?
3. General Knowledge Requirement (0-10): How much external knowledge is needed beyond documents?

## DECISION LOGIC:
- If Document Relevance ≥ 7 AND Document Completeness ≥ 6: Use RAG-REQUIRED intent
- If Document Relevance ≤ 4 AND General Knowledge Requirement ≥ 7: Use GENERAL-KNOWLEDGE intent
- If Document Relevance ≥ 5 AND General Knowledge Requirement ≥ 5: Use HYBRID intent
- If confidence < 0.7: Set requires_clarification = true

## OUTPUT:
Respond with only the intent category name and confidence score (0-1) in this format:
Intent: [category_name]
Confidence: [score]

IMPORTANT: Always respond with valid output format (specified above) only. Do not include any text before or after.
"""
            
            user_prompt = f"""
Analyze the following user query and retrieved documents to determine the appropriate intent:

**User Query:** {query}

**Retrieved Documents:**
{retrieved_docs}

**Additional Context:**
- Average relevance score: {avg_relevance}

Classify the intent and provide your analysis in the required Ouptut format.
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = response.content.strip()
            
            # Parse the response
            intent_match = re.search(r'Intent:\s*(\w+)', result)
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', result)
            
            if intent_match and confidence_match:
                intent_str = intent_match.group(1)
                confidence = float(confidence_match.group(1))
                
                # Map string to IntentType
                try:
                    intent = IntentType(intent_str)
                    return intent, confidence
                except ValueError:
                    # Fallback to general explanation if intent not found
                    return IntentType.EXPLANATIONS, 0.5
            
            # Fallback
            return IntentType.EXPLANATIONS, 0.5
            
        except Exception as e:
            print(f"Error classifying intent: {str(e)}")
            return IntentType.EXPLANATIONS, 0.5
    
    def should_use_rag(self, intent: IntentType) -> bool:
        """
        Determine whether to use RAG or general LLM based on intent.
        
        Args:
            intent: Classified intent
            
        Returns:
            bool: True if should use RAG, False for general LLM
        """
        intent_value = intent.value
        
        if intent_value in GENERAL_INTENTS:
            return False
        else:
            return True
    
    def generate_rag_response(
        self, 
        query: str, 
        rewritten_query: str,
        context_chunks: List[str], 
        chat_history: List[ChatMessage],
        intent: IntentType
    ) -> str:
        """
        Generate response using RAG (context + LLM).
        
        Args:
            query: Original user query
            rewritten_query: Rewritten query
            context_chunks: Retrieved context chunks
            chat_history: Recent chat history
            intent: Classified intent
            
        Returns:
            str: Generated response
        """
        try:
            # Build context from chunks
            context = "\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
            
            # Build chat history context
            history_context = ""
            if chat_history:
                recent_messages = chat_history[-3:]
                for msg in recent_messages:
                    history_context += f"{msg.role}: {msg.content}\n"
            
            # Create RAG prompt based on intent
            system_prompt = f"""You are OrgGPT, an intelligent assistant that helps users find information from their documents. 

Your task is to answer the user's question using the provided context from their documents. Follow these guidelines:

1. Use ONLY the information provided in the context to answer the question
2. If the context doesn't contain enough information, say so clearly
3. Provide specific, accurate, and helpful responses
4. Include relevant details and examples from the context when appropriate
5. Format your response using Markdown for better readability
6. If you reference specific information, indicate which context section it came from

Intent: {intent.value}
This helps you understand what type of information the user is looking for.

Context from Documents:
{context}

Recent Chat History:
{history_context}"""
            
            user_prompt = f"""Original Question: {query}
Rewritten Question: {rewritten_query}

Please provide a comprehensive answer based on the context provided above."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating RAG response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def generate_general_response(
        self, 
        query: str, 
        chat_history: List[ChatMessage],
        intent: IntentType
    ) -> str:
        """
        Generate response using general LLM knowledge.
        
        Args:
            query: User query
            chat_history: Recent chat history
            intent: Classified intent
            
        Returns:
            str: Generated response
        """
        try:
            # Build chat history context
            history_context = ""
            if chat_history:
                recent_messages = chat_history[-3:]

                for msg in recent_messages:
                    history_context += f"{msg.role}: {msg.content}\n"
            
            system_prompt = f"""You are OrgGPT, an intelligent assistant. The user's question doesn't seem to be related to their uploaded documents, so you should provide a helpful general response using your knowledge.

Guidelines:
1. Provide accurate, helpful, and comprehensive information
2. Use your general knowledge to answer the question
3. Format your response using Markdown for better readability
4. Be conversational and friendly
5. If you're not certain about something, acknowledge the uncertainty

Intent: {intent.value}
This helps you understand what type of information the user is looking for.

Recent Chat History:
{history_context}"""
            
            user_prompt = f"Question: {query}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating general response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

