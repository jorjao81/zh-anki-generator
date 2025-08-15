"""Unified field generator for etymology and structural decomposition."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .llm import TokenUsage, FieldGenerationResult
from .ai_config import AIConfigLoader


class BaseFieldGenerator(ABC):
    """Base interface for field generation."""
    
    @abstractmethod 
    def generate_fields(self, entries: List[Any]) -> Dict[int, FieldGenerationResult]:
        """Generate fields for multiple entries."""
        raise NotImplementedError


class StandardFieldGenerator(BaseFieldGenerator):
    """Standard field generator that returns empty results."""
    
    def generate_fields(self, entries: List[Any]) -> Dict[int, FieldGenerationResult]:
        """Return empty results for all entries."""
        return {}


class AIFieldGenerator(BaseFieldGenerator):
    """AI-powered field generator using unified configuration."""
    
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

    def __init__(self, config_loader: Optional[AIConfigLoader] = None):
        """Initialize with configuration."""
        self.config_loader = config_loader or AIConfigLoader()
        self._clients = {}  # Cache for AI clients
        
    def _is_single_character(self, chinese: str) -> bool:
        """Check if the Chinese text is a single character."""
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', chinese)
        return len(chinese_chars) == 1
    
    def _get_char_type(self, chinese: str) -> str:
        """Determine character type for configuration lookup."""
        return "single_char" if self._is_single_character(chinese) else "multi_char"
    
    def _get_client_and_config(self, char_type: str) -> tuple[Any, Dict[str, Any]]:
        """Get AI client and configuration for character type."""
        config = self.config_loader.get_feature_config('field_generation', char_type)
        
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
    
    def _load_prompt(self, prompt_path: str, char_type: str) -> str:
        """Load prompt from file and include examples."""
        full_path = self.config_loader.resolve_prompt_path(prompt_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        
        base_prompt = full_path.read_text(encoding='utf-8')
        
        # Load examples from the appropriate examples directory
        examples_dir = full_path.parent.parent / "examples" / f"field_generation_{char_type}"
        
        if examples_dir.exists():
            return self._load_prompt_with_examples(base_prompt, examples_dir)
        
        return base_prompt
    
    def _load_prompt_with_examples(self, base_prompt: str, examples_dir) -> str:
        """Load examples from the examples directory and incorporate them into the prompt."""
        import glob
        
        examples_text = "\n\nHere are examples of the expected output format:\n\n"
        
        # Check if this is multi-char or single-char based on path
        is_multi_char = "multi_char" in str(examples_dir)
        
        # Load all structural decomposition examples
        structural_files = glob.glob(str(examples_dir / "structural_decomposition*.html"))
        structural_files.sort()  # Ensure consistent ordering
        
        for i, structural_path in enumerate(structural_files, 1):
            with open(structural_path, 'r', encoding='utf-8') as f:
                structural_content = f.read().strip()
                if is_multi_char:
                    example_name = "学校" if i == 1 else f"multi char example {i}"
                else:
                    example_name = "忆" if i == 1 else f"example {i}"  # fallback for shared examples
                examples_text += f"**Example structural_decomposition_html for {example_name}:**\n```html\n{structural_content}\n```\n\n"
        
        # Load all etymology examples
        etymology_files = glob.glob(str(examples_dir / "etymology*.html"))
        etymology_files.sort()  # Ensure consistent ordering
        
        for i, etymology_path in enumerate(etymology_files, 1):
            with open(etymology_path, 'r', encoding='utf-8') as f:
                etymology_content = f.read().strip()
                if is_multi_char:
                    example_name = "学校" if i == 1 else f"multi char example {i}"
                else:
                    example_name = "忆" if i == 1 else f"example {i}"  # fallback for shared examples
                examples_text += f"**Example etymology_html for {example_name}:**\n```html\n{etymology_content}\n```\n\n"
        
        return base_prompt + examples_text
    
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
    
    def _generate_field_for_entry(self, entry: Any) -> Optional[FieldGenerationResult]:
        """Generate fields for a single entry."""
        try:
            chinese = entry.chinese
            pinyin = entry.pinyin
            char_type = self._get_char_type(chinese)
            
            # Check if feature is enabled for this character type
            if not self.config_loader.is_feature_enabled('field_generation', char_type):
                return None
            
            client, config = self._get_client_and_config(char_type)
            provider = config.get('provider', 'gpt')
            
            # Load prompt
            prompt_path = config.get('prompt')
            if not prompt_path:
                raise ValueError(f"No prompt configured for field_generation {char_type}")
            
            prompt = self._load_prompt(prompt_path, char_type)
            
            # Create request data
            request_data = {
                "character": chinese,
                "pinyin": pinyin
            }
            
            user_message = json.dumps(request_data, ensure_ascii=False)
            
            if provider == 'gpt':
                # Handle o1 models differently
                model = config.get('model', 'gpt-4o-mini')
                
                if model.startswith('gpt-5') or model.startswith('o1'):
                    # Use newer parameters for o1 models
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": f"{prompt}\n\n{user_message}"}
                        ],
                        max_completion_tokens=config.get('max_tokens', 800),
                        reasoning_effort=config.get('reasoning_effort', 'medium'),
                    )
                else:
                    # Use standard parameters for other models
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=config.get('temperature', 0.5),
                        max_completion_tokens=config.get('max_tokens', 800),
                    )
                
                response_text = response.choices[0].message.content.strip()
                
                # Calculate token usage
                usage = response.usage
                token_usage = TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=self._calculate_cost(
                        provider, model,
                        usage.prompt_tokens, usage.completion_tokens
                    )
                )
                
            elif provider == 'gemini':
                full_prompt = f"{prompt}\n\n{user_message}"
                
                generation_config = {
                    "temperature": config.get('temperature', 0.5),
                    "response_mime_type": "application/json"
                }
                if 'max_tokens' in config:
                    generation_config["max_output_tokens"] = config['max_tokens']
                
                response = client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                response_text = response.text.strip()
                
                # Extract token usage from response metadata (like original)
                usage_dict = {}
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage_dict = {
                        "prompt_token_count": response.usage_metadata.prompt_token_count,
                        "candidates_token_count": response.usage_metadata.candidates_token_count,
                        "total_token_count": response.usage_metadata.total_token_count,
                    }
                
                token_usage = TokenUsage(
                    prompt_tokens=usage_dict.get("prompt_token_count", 0),
                    completion_tokens=usage_dict.get("candidates_token_count", 0),
                    total_tokens=usage_dict.get("total_token_count", 0),
                    cost_usd=self._calculate_cost(
                        provider, config.get('model'),
                        usage_dict.get("prompt_token_count", 0), usage_dict.get("candidates_token_count", 0)
                    )
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                return FieldGenerationResult(
                    etymology=result.get('etymology_html', ''),
                    structural_decomposition=result.get('structural_decomposition_html', ''),
                    token_usage=token_usage
                )
            except json.JSONDecodeError:
                # Return empty result on parse error
                return FieldGenerationResult(
                    etymology='',
                    structural_decomposition='',
                    token_usage=token_usage
                )
                
        except Exception as e:
            print(f"Field generation error for {entry.chinese}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_fields(self, entries: List[Any]) -> Dict[int, FieldGenerationResult]:
        """Generate fields for multiple entries using ThreadPool."""
        # Filter entries that need field generation
        entries_to_process = []
        for entry in entries:
            char_type = self._get_char_type(entry.chinese)
            if self.config_loader.is_feature_enabled('field_generation', char_type):
                entries_to_process.append(entry)
        
        if not entries_to_process:
            return {}
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_entry = {
                executor.submit(self._generate_field_for_entry, entry): entry
                for entry in entries_to_process
            }
            
            for future in future_to_entry:
                entry = future_to_entry[future]
                try:
                    result = future.result()
                    if result:
                        results[id(entry)] = result
                except Exception as e:
                    # Skip failed entries
                    continue
        
        return results


class UnifiedFieldGeneratorFactory:
    """Factory for creating field generators."""
    
    def __init__(self, config_loader: Optional[AIConfigLoader] = None):
        """Initialize with config loader."""
        self.config_loader = config_loader or AIConfigLoader()
    
    def create_generator(self) -> BaseFieldGenerator:
        """Create appropriate generator based on configuration."""
        try:
            # Check if any character type has field generation enabled
            single_enabled = self.config_loader.is_feature_enabled('field_generation', 'single_char')
            multi_enabled = self.config_loader.is_feature_enabled('field_generation', 'multi_char')
            
            if single_enabled or multi_enabled:
                return AIFieldGenerator(self.config_loader)
            else:
                return StandardFieldGenerator()
                
        except (ValueError, KeyError):
            # Default to standard generator if config issues
            return StandardFieldGenerator()


# Convenience functions for backward compatibility
def create_field_generator(config_loader: Optional[AIConfigLoader] = None) -> BaseFieldGenerator:
    """Create field generator using unified configuration."""
    factory = UnifiedFieldGeneratorFactory(config_loader)
    return factory.create_generator()