from typing import Optional, Dict, Any
import logging
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_models import ChatAnthropic
from langchain.schema.language_model import BaseLanguageModel

from app.config.settings import get_settings, LLMProvider

logger = logging.getLogger(__name__)

class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    def __init__(self):
        self.settings = get_settings()
        self._clients: Dict[str, BaseLanguageModel] = {}
    
    def get_llm(self, model_name: str) -> BaseLanguageModel:
        """Get or create LLM client"""
        if model_name in self._clients:
            return self._clients[model_name]
        
        client = self._create_llm_client(model_name)
        self._clients[model_name] = client
        return client
    
    def _create_llm_client(self, model_name: str) -> BaseLanguageModel:
        """Create LLM client based on model name"""
        
        # OpenAI models
        if model_name.startswith(('gpt-', 'openai')):
            if not self.settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            
            return ChatOpenAI(
                model=model_name.replace('openai_', ''),
                api_key=self.settings.openai_api_key,
                temperature=0.0,
                max_tokens=4000
            )
        
        # Anthropic models
        elif model_name.startswith(('claude', 'anthropic')):
            if not self.settings.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            
            return ChatAnthropic(
                model=model_name.replace('anthropic_', ''),
                anthropic_api_key=self.settings.anthropic_api_key,
                temperature=0.0,
                max_tokens=4000
            )
        
        # Google models
        elif model_name.startswith(('gemini', 'google')):
            if not self.settings.gemini_api_key:
                raise ValueError("Gemini API key not configured")
            
            return ChatVertexAI(
                model_name=model_name.replace('google_', ''),
                temperature=0.0,
                max_output_tokens=4000
            )
        
        # Azure OpenAI
        elif model_name.startswith('azure'):
            if not self.settings.azure_openai_key or not self.settings.azure_openai_endpoint:
                raise ValueError("Azure OpenAI credentials not configured")
            
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_deployment=model_name.replace('azure_', ''),
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_key,
                api_version="2024-02-15-preview",
                temperature=0.0
            )
        
        else:
            # Default to OpenAI if model not recognized
            logger.warning(f"Unknown model {model_name}, defaulting to OpenAI")
            return ChatOpenAI(
                model=self.settings.default_llm_model,
                api_key=self.settings.openai_api_key,
                temperature=0.0
            )
    
    def get_available_models(self) -> Dict[str, list]:
        """Get available models by provider"""
        available = {}
        
        if self.settings.openai_api_key:
            available[LLMProvider.OPENAI] = [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
            ]
        
        if self.settings.anthropic_api_key:
            available[LLMProvider.ANTHROPIC] = [
                "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"
            ]
        
        if self.settings.gemini_api_key:
            available[LLMProvider.GEMINI] = [
                "gemini-1.5-pro", "gemini-1.5-flash"
            ]
        
        return available
