"""Unified AI configuration loader and manager."""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class FeatureConfig(BaseModel):
    """Configuration for an AI feature."""
    provider: str = "standard"  # "standard", "gpt", or "gemini"
    model: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    use_web_search: Optional[bool] = None
    use_structured_output: Optional[bool] = None
    fallback_models: Optional[List[Dict[str, Any]]] = None


class CharTypeConfig(BaseModel):
    """Configuration for single/multi-character specific settings."""
    provider: str = "standard"  # "standard", "gpt", or "gemini"
    model: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    use_web_search: Optional[bool] = None


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
        
        return self._config
    
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
    
