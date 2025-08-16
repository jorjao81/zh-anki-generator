"""Unified AI configuration loader and manager."""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, validator
import logging


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class FeatureConfig(BaseModel):
    """Configuration for an AI feature."""
    provider: str = "standard"  # "standard", "gpt", "gemini", or "deepseek"
    model: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    use_web_search: Optional[bool] = None
    use_structured_output: Optional[bool] = None
    fallback_models: Optional[List[Dict[str, Any]]] = None
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = ["standard", "gpt", "gemini", "deepseek"]
        if v not in valid_providers:
            raise ValueError(f"Invalid provider '{v}'. Must be one of: {valid_providers}")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError(f"Temperature must be between 0 and 2, got {v}")
        return v


class CharTypeConfig(BaseModel):
    """Configuration for single/multi-character specific settings."""
    provider: str = "standard"  # "standard", "gpt", "gemini", or "deepseek"
    model: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    use_web_search: Optional[bool] = None
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = ["standard", "gpt", "gemini", "deepseek"]
        if v not in valid_providers:
            raise ValueError(f"Invalid provider '{v}'. Must be one of: {valid_providers}")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError(f"Temperature must be between 0 and 2, got {v}")
        return v


class FieldGenerationConfig(BaseModel):
    """Configuration for field generation feature."""
    single_char: CharTypeConfig = CharTypeConfig()
    multi_char: CharTypeConfig = CharTypeConfig()


class FormatterConfig(BaseModel):
    """Configuration for formatter features."""
    single_char: CharTypeConfig = CharTypeConfig()
    multi_char: CharTypeConfig = CharTypeConfig()


class GlobalConfig(BaseModel):
    """Global AI configuration."""
    default_provider: str = "gpt"
    max_daily_cost_usd: Optional[float] = None
    warn_at_cost_usd: Optional[float] = None


class AIConfig(BaseModel):
    """Complete AI configuration."""
    global_config: GlobalConfig = GlobalConfig()
    providers: Dict[str, ProviderConfig] = {}
    features: Dict[str, Union[FeatureConfig, FieldGenerationConfig, FormatterConfig]] = {}


class AIConfigLoader:
    """Loads and manages AI configuration."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize with config file path."""
        if config_path is None:
            # Default to ai_config/ai_config.yaml in project root
            config_path = Path(__file__).parent.parent.parent / "ai_config" / "ai_config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Optional[AIConfig] = None
        
    def load_config(self) -> AIConfig:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config
            
        if not self.config_path.exists():
            raise FileNotFoundError(f"AI config file not found: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
            
        # Parse configuration sections
        global_config = GlobalConfig(**(raw_config.get('global', {})))
        
        providers = {}
        for name, config in raw_config.get('providers', {}).items():
            providers[name] = ProviderConfig(**config)
        
        features = {}
        for name, config in raw_config.get('features', {}).items():
            if name == 'field_generation':
                features[name] = FieldGenerationConfig(**config)
            elif name in ['meaning_formatter', 'examples_formatter']:
                features[name] = FormatterConfig(**config)
            else:
                features[name] = FeatureConfig(**config)
        
        self._config = AIConfig(
            global_config=global_config,
            providers=providers,
            features=features
        )
        
        # Validate the configuration
        self.validate_config()
        
        return self._config
    
    def validate_config(self):
        """Validate the entire configuration including model names and API keys."""
        if self._config is None:
            raise ValueError("Configuration not loaded")
        
        logger = logging.getLogger(__name__)
        errors = []
        warnings = []
        
        # Validate model names and API keys for enabled features
        for feature_name, feature_config in self._config.features.items():
            try:
                if hasattr(feature_config, 'single_char'):
                    # Field generation or formatter config
                    for char_type in ['single_char', 'multi_char']:
                        char_config = getattr(feature_config, char_type)
                        if char_config.provider != 'standard':
                            self._validate_feature_provider(feature_name, char_type, char_config, errors, warnings)
                else:
                    # Simple feature config (like word_classifier)
                    if feature_config.provider != 'standard':
                        self._validate_feature_provider(feature_name, None, feature_config, errors, warnings)
            except Exception as e:
                errors.append(f"Error validating feature '{feature_name}': {str(e)}")
        
        # Validate prompt files exist
        self._validate_prompt_files(errors)
        
        # Report results
        if warnings:
            logger.warning("AI Configuration warnings:")
            for warning in warnings:
                logger.warning(f"  • {warning}")
        
        if errors or warnings:  # FAIL FAST on any validation issues
            all_issues = errors + [f"Warning: {w}" for w in warnings]
            error_msg = "AI Configuration validation failed:\n" + "\n".join(f"  • {issue}" for issue in all_issues)
            raise ValueError(error_msg)
        
        logger.info("✅ AI configuration validation passed")
    
    def _validate_feature_provider(self, feature_name: str, char_type: Optional[str], config, errors: List[str], warnings: List[str]):
        """Validate a specific feature's provider configuration."""
        provider = config.provider
        model = config.model
        
        feature_desc = f"{feature_name}" + (f".{char_type}" if char_type else "")
        
        # Validate model names
        if model:
            if provider == 'gpt':
                valid_gpt_models = [
                    # GPT-5 models
                    'gpt-5', 'gpt-5-nano', 'gpt-5-mini', 'gpt-5-chat-latest',
                    'gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07',
                    # GPT-4 models  
                    'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo',
                    'gpt-4o-2024-11-20', 'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18'
                ]
                if model not in valid_gpt_models:
                    warnings.append(f"{feature_desc}: Unknown GPT model '{model}'. Known models: {', '.join(valid_gpt_models[:5])}...")
            elif provider == 'gemini':
                valid_gemini_models = [
                    'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
                    'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro'
                ]
                if model not in valid_gemini_models:
                    warnings.append(f"{feature_desc}: Unknown Gemini model '{model}'. Known models: {', '.join(valid_gemini_models)}")
            elif provider == 'deepseek':
                valid_deepseek_models = [
                    'deepseek-chat', 'deepseek-reasoner'
                ]
                if model not in valid_deepseek_models:
                    warnings.append(f"{feature_desc}: Unknown DeepSeek model '{model}'. Known models: {', '.join(valid_deepseek_models)}")
        else:
            warnings.append(f"{feature_desc}: No model specified for provider '{provider}'")
        
        # Validate API keys
        if provider == 'gpt':
            api_key = self._get_api_key_for_provider('gpt')
            if not api_key:
                errors.append(f"{feature_desc}: OpenAI API key not found. Set OPENAI_API_KEY environment variable or specify in providers.gpt.api_key")
            else:
                self._test_api_key('gpt', api_key, feature_desc, errors, warnings)
        elif provider == 'gemini':
            api_key = self._get_api_key_for_provider('gemini')
            if not api_key:
                errors.append(f"{feature_desc}: Gemini API key not found. Set GEMINI_API_KEY environment variable or specify in providers.gemini.api_key")
            else:
                self._test_api_key('gemini', api_key, feature_desc, errors, warnings)
        elif provider == 'deepseek':
            api_key = self._get_api_key_for_provider('deepseek')
            if not api_key:
                errors.append(f"{feature_desc}: DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable or specify in providers.deepseek.api_key")
            else:
                self._test_api_key('deepseek', api_key, feature_desc, errors, warnings)
    
    def _get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        # Check config first
        if provider in self._config.providers:
            provider_config = self._config.providers[provider]
            if provider_config.api_key:
                return provider_config.api_key
        
        # Check environment variables
        if provider == 'gpt':
            return os.getenv('OPENAI_API_KEY')
        elif provider == 'gemini':
            return os.getenv('GEMINI_API_KEY')
        elif provider == 'deepseek':
            return os.getenv('DEEPSEEK_API_KEY')
        
        return None
    
    def _test_api_key(self, provider: str, api_key: str, feature_desc: str, errors: List[str], warnings: List[str]):
        """Test API key by making a simple request."""
        try:
            if provider == 'gpt':
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                # Simple test request
                client.models.list()
                
            elif provider == 'gemini':
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                # Simple test request
                list(genai.list_models())
                
            elif provider == 'deepseek':
                from openai import OpenAI
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
                # Simple test request
                client.models.list()
                
        except Exception as e:
            error_msg = str(e)
            if "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
                errors.append(f"{feature_desc}: Invalid {provider.upper()} API key")
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                warnings.append(f"{feature_desc}: {provider.upper()} API quota/rate limit reached")
            else:
                warnings.append(f"{feature_desc}: Could not verify {provider.upper()} API key: {error_msg}")
    
    def _validate_prompt_files(self, errors: List[str]):
        """Validate that all prompt files exist."""
        for feature_name, feature_config in self._config.features.items():
            if hasattr(feature_config, 'single_char'):
                # Field generation or formatter config
                for char_type in ['single_char', 'multi_char']:
                    char_config = getattr(feature_config, char_type)
                    if char_config.prompt:
                        prompt_path = self.resolve_prompt_path(char_config.prompt)
                        if not prompt_path.exists():
                            errors.append(f"{feature_name}.{char_type}: Prompt file not found: {prompt_path}")
            else:
                # Simple feature config
                if feature_config.prompt:
                    prompt_path = self.resolve_prompt_path(feature_config.prompt)
                    if not prompt_path.exists():
                        errors.append(f"{feature_name}: Prompt file not found: {prompt_path}")
    
    def get_feature_config(self, feature_name: str, char_type: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific feature and character type."""
        config = self.load_config()
        
        if feature_name not in config.features:
            raise ValueError(f"Feature '{feature_name}' not found in configuration")
        
        feature_config = config.features[feature_name]
        
        # Handle different config types
        if char_type is not None and hasattr(feature_config, char_type):
            # For features with single_char/multi_char sub-configs
            char_config = getattr(feature_config, char_type)
            base_config = feature_config.model_dump()
            char_config_dict = char_config.model_dump()
            
            # Merge base config with character-specific config
            result = {**base_config}
            result.update({k: v for k, v in char_config_dict.items() if v is not None})
        else:
            # For simple feature configs
            result = feature_config.model_dump()
        
        # Apply defaults from global config
        if 'provider' not in result or result['provider'] is None:
            result['provider'] = config.global_config.default_provider
            
        # Resolve provider settings
        provider_name = result.get('provider')
        if provider_name and provider_name in config.providers:
            provider_config = config.providers[provider_name]
            result['api_key'] = result.get('api_key') or provider_config.api_key
            result['base_url'] = result.get('base_url') or provider_config.base_url
            
        # Set API key from environment if not specified
        if result.get('api_key') is None:
            if provider_name == 'gpt':
                result['api_key'] = os.getenv('OPENAI_API_KEY')
            elif provider_name == 'gemini':
                result['api_key'] = os.getenv('GEMINI_API_KEY')
            elif provider_name == 'deepseek':
                result['api_key'] = os.getenv('DEEPSEEK_API_KEY')
        
        return result
    
    
    def is_feature_enabled(self, feature_name: str, char_type: Optional[str] = None) -> bool:
        """Check if a feature is enabled (not using 'standard' provider)."""
        try:
            config_dict = self.get_feature_config(feature_name, char_type)
            return config_dict.get('provider', 'standard') != 'standard'
        except (ValueError, KeyError):
            return False
    
    def resolve_prompt_path(self, prompt_path: str) -> Path:
        """Resolve prompt path relative to project root."""
        if Path(prompt_path).is_absolute():
            return Path(prompt_path)
        else:
            # Relative to project root
            project_root = Path(__file__).parent.parent.parent
            return project_root / prompt_path
    
