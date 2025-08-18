"""Unified word classifier using AI configuration."""

from __future__ import annotations

import json
from typing import Optional, NamedTuple, List
from pydantic import BaseModel

from .ai_config import AIConfigLoader


class WordClassification(NamedTuple):
    """Result of word classification."""
    word: str
    definition: str  # Short English definition
    classification: str  # "worth_learning", "compositional", "not_a_word", "proper_name"


class UnifiedWordClassifier:
    """Word classifier using unified AI configuration."""
    
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

    # DeepSeek pricing per 1M tokens
    DEEPSEEK_PRICING = {
        "deepseek-chat": {"input_cache_hit": 0.07, "input_cache_miss": 0.27, "output": 1.10},
        "deepseek-reasoner": {"input_cache_hit": 0.14, "input_cache_miss": 0.55, "output": 2.19},
    }

    def __init__(self, config_loader: Optional[AIConfigLoader] = None):
        """Initialize classifier with configuration."""
        self.config_loader = config_loader or AIConfigLoader()
        self._client = None
        self._config = None
    
    def _initialize_client(self):
        """Initialize AI client if not already done."""
        if self._client is not None:
            return
        
        self._config = self.config_loader.get_feature_config('word_classifier')
        provider = self._config.get('provider', 'gpt')
        
        # Get provider-level configuration for API keys
        providers_dict = self.config_loader.load_config().providers
        provider_config = providers_dict.get(provider) if provider in providers_dict else None
        
        if provider == 'gpt':
            from openai import OpenAI
            import os
            api_key = self._config.get('api_key') or (provider_config.api_key if provider_config else None) or os.getenv('OPENAI_API_KEY')
            base_url = self._config.get('base_url') or (provider_config.base_url if provider_config else None)
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        elif provider == 'gemini':
            import google.generativeai as genai
            import os
            api_key = self._config.get('api_key') or (provider_config.api_key if provider_config else None) or os.getenv('GEMINI_API_KEY')
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self._config.get('model'))
        elif provider == 'deepseek':
            from openai import OpenAI
            import os
            api_key = self._config.get('api_key') or (provider_config.api_key if provider_config else None) or os.getenv('DEEPSEEK_API_KEY')
            base_url = self._config.get('base_url') or (provider_config.base_url if provider_config else None) or 'https://api.deepseek.com'
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _load_prompt(self) -> str:
        """Load classification prompt."""
        prompt_path = self._config.get('prompt')
        if not prompt_path:
            raise ValueError("No prompt configured for word_classifier")
        
        full_path = self.config_loader.resolve_prompt_path(prompt_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        
        return full_path.read_text(encoding='utf-8')
    
    def _calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        if provider == 'gpt':
            pricing = self.GPT_PRICING.get(model, {"input": 0.15, "output": 0.60})
            input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
            output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
        elif provider == 'gemini':
            pricing = self.GEMINI_PRICING.get(model, {"input": 0.075, "output": 0.30})
            input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
            output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
        elif provider == 'deepseek':
            pricing = self.DEEPSEEK_PRICING.get(model, {"input_cache_miss": 0.27, "output": 1.10})
            # Assume cache miss for simplicity
            input_cost = (input_tokens / 1_000_000) * pricing.get("input_cache_miss", 0)
            output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
        else:
            return 0.0
        
        return input_cost + output_cost
    
    def is_available(self) -> bool:
        """Check if classifier is available and configured."""
        try:
            return self.config_loader.is_feature_enabled('word_classifier')
        except (ValueError, KeyError):
            return False
    
    def classify_word(self, word: str) -> Optional[WordClassification]:
        """Classify a single word with fallback models."""
        if not self.is_available():
            return None
        
        # Initialize client and config if needed
        self._initialize_client()
        
        # Try primary model first
        result = self._classify_with_config(word, self._config)
        
        # If result is unknown, try fallback models
        if result and result.classification == 'unknown':
            fallback_configs = self._config.get('fallback_models', []) or []
            for fallback_config in fallback_configs:
                try:
                    result = self._classify_with_config(word, fallback_config)
                    if result and result.classification != 'unknown':
                        break
                except Exception:
                    continue
        
        return result
    
    def _classify_with_config(self, word: str, config: dict) -> Optional[WordClassification]:
        """Classify word with specific configuration."""
        try:
            provider = config.get('provider', 'gpt')
            
            # Initialize client for this config if needed
            providers_dict = self.config_loader.load_config().providers
            provider_config = providers_dict.get(provider) if provider in providers_dict else None
            
            if provider == 'gpt':
                from openai import OpenAI
                import os
                api_key = (config.get('api_key') or 
                          self._config.get('api_key') or 
                          (provider_config.api_key if provider_config else None) or 
                          os.getenv('OPENAI_API_KEY'))
                base_url = (config.get('base_url') or 
                           self._config.get('base_url') or 
                           (provider_config.base_url if provider_config else None))
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            elif provider == 'gemini':
                import google.generativeai as genai
                import os
                api_key = (config.get('api_key') or 
                          self._config.get('api_key') or 
                          (provider_config.api_key if provider_config else None) or 
                          os.getenv('GEMINI_API_KEY'))
                genai.configure(api_key=api_key)
                client = genai.GenerativeModel(config.get('model'))
            elif provider == 'deepseek':
                from openai import OpenAI
                import os
                api_key = (config.get('api_key') or 
                          self._config.get('api_key') or 
                          (provider_config.api_key if provider_config else None) or 
                          os.getenv('DEEPSEEK_API_KEY'))
                base_url = (config.get('base_url') or 
                           self._config.get('base_url') or 
                           (provider_config.base_url if provider_config else None) or 
                           'https://api.deepseek.com')
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            else:
                return None
            
            prompt = self._load_prompt()
            
            if provider in ['gpt', 'deepseek']:
                # Handle o1 models differently (GPT only)
                model = config.get('model', 'gpt-4o-mini' if provider == 'gpt' else 'deepseek-chat')
                
                if provider == 'gpt' and (model.startswith('gpt-5') or model.startswith('o1')):
                    # Use newer parameters for o1 models
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": f"{prompt}\n\nWord to classify: {word}"}
                        ],
                        max_completion_tokens=config.get('max_tokens', 200),
                        reasoning_effort=config.get('reasoning_effort', 'medium'),
                    )
                else:
                    # Use structured output if enabled
                    if config.get('use_structured_output', True):
                        if provider == 'deepseek':
                            # DeepSeek uses json_object mode, not json_schema
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": word}
                                ],
                                max_completion_tokens=config.get('max_tokens', 200),
                                response_format={"type": "json_object"}
                            )
                        else:
                            # GPT uses json_schema mode
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": word}
                                ],
                                max_completion_tokens=config.get('max_tokens', 200),
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "word_classification",
                                        "strict": True,
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "definition": {"type": "string"},
                                                "classification": {
                                                    "type": "string",
                                                    "enum": ["worth_learning", "compositional", "not_a_word", "proper_name"]
                                                }
                                            },
                                            "required": ["definition", "classification"],
                                            "additionalProperties": False
                                        }
                                    }
                                }
                            )
                    else:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": word}
                            ],
                            temperature=config.get('temperature', 0.3),
                            max_completion_tokens=config.get('max_tokens', 200),
                        )
                
                response_text = response.choices[0].message.content.strip()
                
            elif provider == 'gemini':
                full_prompt = f"{prompt}\n\nWord to classify: {word}"
                
                generation_config = {
                    "temperature": config.get('temperature', 0.3),
                    "max_output_tokens": config.get('max_tokens', 200),
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "definition": {"type": "string"},
                            "classification": {
                                "type": "string",
                                "enum": ["worth_learning", "compositional", "not_a_word", "proper_name"]
                            }
                        },
                        "required": ["definition", "classification"]
                    }
                }
                
                response = client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                response_text = response.text.strip()
            else:
                return None
            
            # Parse JSON response
            try:
                # Clean up response text - remove markdown code blocks if present
                clean_text = response_text.strip()
                if clean_text.startswith('```json'):
                    clean_text = clean_text[7:]  # Remove ```json
                if clean_text.startswith('```'):
                    clean_text = clean_text[3:]  # Remove ```
                if clean_text.endswith('```'):
                    clean_text = clean_text[:-3]  # Remove ```
                clean_text = clean_text.strip()
                
                result = json.loads(clean_text)
                return WordClassification(
                    word=word,
                    definition=result.get('definition', 'unknown'),
                    classification=result.get('classification', 'unknown')
                )
            except json.JSONDecodeError as e:
                # Fallback parsing if JSON is malformed
                import logging
                logging.error(f"JSON parse error for word '{word}': {e}")
                logging.error(f"Response text: '{response_text}'")
                logging.error(f"Response length: {len(response_text)}")
                return WordClassification(
                    word=word,
                    definition='parse_error',
                    classification='unknown'
                )
                
        except Exception as e:
            return WordClassification(
                word=word,
                definition='error',
                classification='unknown'
            )
    
    def classify_words(self, words: List[str]) -> List[WordClassification]:
        """Classify multiple words."""
        results = []
        for word in words:
            classification = self.classify_word(word)
            if classification:
                results.append(classification)
        return results
    
    def classify_words_batch(self, words: List[str], max_workers: int = 3) -> List[WordClassification]:
        """Classify multiple words with threading for better performance."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if not self.is_available():
            return []
        
        # Initialize client once before threading
        self._initialize_client()
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all classification tasks
            future_to_word = {
                executor.submit(self.classify_word, word): word 
                for word in words
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_word):
                word = future_to_word[future]
                try:
                    classification = future.result()
                    if classification:
                        results.append(classification)
                except Exception as e:
                    # Create error classification for failed words
                    results.append(WordClassification(
                        word=word,
                        definition='error',
                        classification='unknown'
                    ))
        
        # Sort results to maintain original word order
        word_order = {word: i for i, word in enumerate(words)}
        results.sort(key=lambda x: word_order.get(x.word, len(words)))
        
        return results


# Factory function for backward compatibility
def create_word_classifier(config_loader: Optional[AIConfigLoader] = None) -> UnifiedWordClassifier:
    """Create word classifier using unified configuration."""
    return UnifiedWordClassifier(config_loader)