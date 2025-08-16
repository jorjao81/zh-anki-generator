"""Unified AI-powered field formatting for Anki cards."""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel

from .llm import TokenUsage
from .ai_config import AIConfigLoader

logger = logging.getLogger(__name__)


class FieldFormattingResult(BaseModel):
    """Result from field formatting operation."""
    
    formatted_content: str
    token_usage: Optional[TokenUsage] = None


class BaseFieldFormatter(ABC):
    """Base interface for field formatting."""
    
    @abstractmethod
    def format_field(self, hanzi: str, original_content: str) -> FieldFormattingResult:
        """Format a field using the configured method."""
        raise NotImplementedError


class StandardFieldFormatter(BaseFieldFormatter):
    """Standard field formatter that returns content unchanged."""
    
    def format_field(self, hanzi: str, original_content: str) -> FieldFormattingResult:
        """Return content unchanged."""
        return FieldFormattingResult(formatted_content=original_content)


class AIFieldFormatter(BaseFieldFormatter):
    """AI-powered field formatter using unified configuration."""
    
    # GPT pricing per 1M tokens
    GPT_PRICING = {
        "gpt-5": {"input": 1.25, "output": 10.00, "cached_input": 0.125},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    }

    # Gemini pricing per 1M tokens
    GEMINI_PRICING = {
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    }

    def __init__(self, feature_name: str, config_loader: Optional[AIConfigLoader] = None):
        """Initialize formatter with feature configuration."""
        self.feature_name = feature_name
        self.config_loader = config_loader or AIConfigLoader()
        self._clients = {}  # Cache for AI clients
        
    def _is_single_character(self, chinese: str) -> bool:
        """Check if the Chinese text is a single character."""
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', chinese)
        return len(chinese_chars) == 1
    
    def _get_char_type(self, hanzi: str) -> str:
        """Determine character type for configuration lookup."""
        return "single_char" if self._is_single_character(hanzi) else "multi_char"
    
    def _get_client_and_config(self, hanzi: str) -> tuple[Any, Dict[str, Any]]:
        """Get AI client and configuration for the given text."""
        char_type = self._get_char_type(hanzi)
        config = self.config_loader.get_feature_config(self.feature_name, char_type)
        
        provider = config.get('provider', 'gpt')
        client_key = f"{provider}_{char_type}"
        
        if client_key not in self._clients:
            if provider == 'gpt':
                from openai import OpenAI
                self._clients[client_key] = OpenAI(
                    api_key=config.get('api_key'),
                    base_url=config.get('base_url')
                )
            elif provider == 'gemini':
                import google.generativeai as genai
                genai.configure(api_key=config.get('api_key'))
                self._clients[client_key] = genai.GenerativeModel(config.get('model'))
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        return self._clients[client_key], config
    
    def _load_prompt(self, prompt_path: str) -> str:
        """Load prompt from file."""
        full_path = self.config_loader.resolve_prompt_path(prompt_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        
        return full_path.read_text(encoding='utf-8')
    
    def _calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        if provider == 'gpt':
            pricing = self.GPT_PRICING.get(model, {"input": 0.15, "output": 0.60})
        elif provider == 'gemini':
            pricing = self.GEMINI_PRICING.get(model, {"input": 0.075, "output": 0.30})
        else:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
        
        return input_cost + output_cost
    
    def format_field(self, hanzi: str, original_content: str) -> FieldFormattingResult:
        """Format field using AI."""
        char_type = self._get_char_type(hanzi)
        
        # Check if feature is enabled for this character type
        enabled = self.config_loader.is_feature_enabled(self.feature_name, char_type)
        
        if not enabled:
            return FieldFormattingResult(formatted_content=original_content)
        
        try:
            client, config = self._get_client_and_config(hanzi)
            provider = config.get('provider', 'gpt')
            
            # Load and format prompt
            prompt_path = config.get('prompt')
            if not prompt_path:
                raise ValueError(f"No prompt configured for {self.feature_name} {char_type}")
            
            prompt_template = self._load_prompt(prompt_path)
            
            # Create context for formatting
            user_message = f"Chinese word/character: {hanzi}\nOriginal content: {original_content}"
            
            if provider == 'gpt':
                response = client.chat.completions.create(
                    model=config.get('model', 'gpt-4o-mini'),
                    messages=[
                        {"role": "system", "content": prompt_template},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=config.get('temperature', 0.3),
                    max_completion_tokens=config.get('max_tokens', 400),
                )
                
                formatted_content = response.choices[0].message.content.strip()
                
                # Calculate token usage
                usage = response.usage
                token_usage = TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=self._calculate_cost(
                        provider, config.get('model'), 
                        usage.prompt_tokens, usage.completion_tokens
                    )
                )
                
            elif provider == 'gemini':
                full_prompt = f"{prompt_template}\n\n{user_message}"
                
                generation_config = {
                    "temperature": config.get('temperature', 0.3),
                    "max_output_tokens": config.get('max_tokens', 400),
                }
                
                response = client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                formatted_content = response.text.strip()
                
                # Estimate token usage for Gemini (approximate)
                estimated_input_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
                estimated_output_tokens = len(formatted_content.split()) * 1.3
                
                token_usage = TokenUsage(
                    prompt_tokens=int(estimated_input_tokens),
                    completion_tokens=int(estimated_output_tokens),
                    total_tokens=int(estimated_input_tokens + estimated_output_tokens),
                    cost_usd=self._calculate_cost(
                        provider, config.get('model'),
                        int(estimated_input_tokens), int(estimated_output_tokens)
                    )
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            return FieldFormattingResult(
                formatted_content=formatted_content,
                token_usage=token_usage
            )
            
        except Exception as e:
            # Return original content on error
            logger.error(f"Error in {self.feature_name} formatter: {str(e)}")
            return FieldFormattingResult(formatted_content=original_content)


class UnifiedFieldFormatterFactory:
    """Factory for creating field formatters."""
    
    def __init__(self, config_loader: Optional[AIConfigLoader] = None):
        """Initialize with config loader."""
        self.config_loader = config_loader or AIConfigLoader()
    
    def create_formatter(self, feature_name: str) -> BaseFieldFormatter:
        """Create appropriate formatter based on configuration."""
        try:
            # Check if any character type for this feature is AI-enabled
            for char_type in ['single_char', 'multi_char']:
                if self.config_loader.is_feature_enabled(feature_name, char_type):
                    return AIFieldFormatter(feature_name, self.config_loader)
            
            # If no character types are AI-enabled, use standard formatter
            return StandardFieldFormatter()
                
        except (ValueError, KeyError):
            # Default to standard formatter if config issues
            return StandardFieldFormatter()


# Convenience functions for backward compatibility
def create_meaning_formatter(config_loader: Optional[AIConfigLoader] = None) -> BaseFieldFormatter:
    """Create meaning formatter using unified configuration."""
    factory = UnifiedFieldFormatterFactory(config_loader)
    return factory.create_formatter('meaning_formatter')


def create_examples_formatter(config_loader: Optional[AIConfigLoader] = None) -> BaseFieldFormatter:
    """Create examples formatter using unified configuration."""
    factory = UnifiedFieldFormatterFactory(config_loader)
    return factory.create_formatter('examples_formatter')