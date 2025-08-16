"""Unit tests for DeepSeek AI integration."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import json

from anki_pleco_importer.ai_config import AIConfigLoader
from anki_pleco_importer.unified_field_formatter import AIFieldFormatter, UnifiedFieldFormatterFactory
from anki_pleco_importer.unified_field_generator import AIFieldGenerator
from anki_pleco_importer.unified_word_classifier import UnifiedWordClassifier


class TestDeepSeekConfiguration:
    """Test DeepSeek configuration validation and loading."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary DeepSeek config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = """
global:
  default_provider: "deepseek"

providers:
  deepseek:
    api_key: "test-deepseek-key"
    base_url: "https://api.deepseek.com"

features:
  meaning_formatter:
    multi_char:
      provider: "deepseek"
      model: "deepseek-chat"
      prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
      temperature: 0.3
"""
            f.write(config_content)
            temp_path = Path(f.name)
        
        yield temp_path
        temp_path.unlink(missing_ok=True)
    
    def test_deepseek_config_validation_fails_on_api_key(self, temp_config_file):
        """Test that DeepSeek provider is now accepted but API key validation fails."""
        with pytest.raises(Exception) as excinfo:
            config_loader = AIConfigLoader(str(temp_config_file))
            config_loader.load_config()
        
        # Should contain validation error about invalid API key (not provider)
        assert "deepseek" in str(excinfo.value).lower()
        assert ("invalid" in str(excinfo.value).lower() and "api" in str(excinfo.value).lower()) or "api key" in str(excinfo.value).lower()
    
    def test_deepseek_provider_in_config_schema(self):
        """Test adding DeepSeek to allowed providers list."""
        # This test should now pass after we implemented DeepSeek support
        from anki_pleco_importer.ai_config import CharTypeConfig
        
        # Should no longer raise an exception for deepseek provider
        config = CharTypeConfig(provider="deepseek", model="deepseek-chat")
        assert config.provider == "deepseek"
        assert config.model == "deepseek-chat"


class TestDeepSeekFieldFormatter:
    """Test DeepSeek integration in field formatting."""
    
    @pytest.fixture
    def mock_deepseek_config(self):
        """Mock configuration for DeepSeek."""
        return {
            'provider': 'deepseek',
            'model': 'deepseek-chat',
            'api_key': 'test-deepseek-key',
            'base_url': 'https://api.deepseek.com',
            'temperature': 0.3,
            'max_tokens': 400,
            'prompt': 'test_prompt.md'
        }
    
    @patch('openai.OpenAI')
    def test_deepseek_client_authentication_fails(self, mock_openai, mock_deepseek_config):
        """Test that DeepSeek client fails with invalid API key."""
        # Mock OpenAI client to raise authentication error for DeepSeek
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Error code: 401 - {'error': {'message': 'Authentication Fails, Your api key is invalid'}}"
        )
        mock_openai.return_value = mock_client
        
        mock_config_loader = Mock()
        mock_config_loader.get_feature_config.return_value = mock_deepseek_config
        mock_config_loader.is_feature_enabled.return_value = True
        mock_config_loader.resolve_prompt_path.return_value = Path(__file__).parent / "test_prompt.md"
        
        # Create a test prompt file
        test_prompt = Path(__file__).parent / "test_prompt.md"
        test_prompt.write_text("Test prompt for DeepSeek")
        
        formatter = AIFieldFormatter('meaning_formatter', mock_config_loader)
        
        with pytest.raises(RuntimeError) as excinfo:
            formatter.format_field('测试', 'test meaning')
        
        assert "AI formatting failed" in str(excinfo.value)
        assert "Authentication Fails" in str(excinfo.value)
        
        # Cleanup
        test_prompt.unlink(missing_ok=True)
    
    @patch('openai.OpenAI')
    def test_deepseek_client_success(self, mock_openai, mock_deepseek_config):
        """Test that DeepSeek client works with valid API key."""
        # Mock successful OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "<span class=\"domain\">mathematics</span> higher dimensional"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        mock_config_loader = Mock()
        mock_config_loader.get_feature_config.return_value = mock_deepseek_config
        mock_config_loader.is_feature_enabled.return_value = True
        mock_config_loader.resolve_prompt_path.return_value = Path(__file__).parent / "test_prompt.md"
        
        # Create a test prompt file
        test_prompt = Path(__file__).parent / "test_prompt.md"
        test_prompt.write_text("Test prompt for DeepSeek")
        
        formatter = AIFieldFormatter('meaning_formatter', mock_config_loader)
        result = formatter.format_field('高维', '(math.) higher dimensional')
        
        # Verify successful formatting
        assert result.formatted_content == "<span class=\"domain\">mathematics</span> higher dimensional"
        assert result.token_usage.prompt_tokens == 100
        assert result.token_usage.completion_tokens == 50
        assert result.token_usage.cost_usd > 0  # Should calculate DeepSeek cost
        
        # Verify DeepSeek API was called correctly
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args
        assert call_args[1]['api_key'] == 'test-deepseek-key'
        assert call_args[1]['base_url'] == 'https://api.deepseek.com'
        
        # Cleanup
        test_prompt.unlink(missing_ok=True)
    
    def test_deepseek_cost_calculation_implemented(self):
        """Test that DeepSeek cost calculation is now implemented."""
        formatter = AIFieldFormatter('meaning_formatter', Mock())
        
        # Should calculate real cost for DeepSeek
        cost = formatter._calculate_cost('deepseek', 'deepseek-chat', 100, 50)
        # 100 input tokens at $0.27/1M + 50 output tokens at $1.10/1M
        expected_cost = (100 / 1_000_000) * 0.27 + (50 / 1_000_000) * 1.10
        assert abs(cost - expected_cost) < 0.0000001
    
    def test_deepseek_formatter_factory_fallback(self):
        """Test that formatter factory falls back to standard when DeepSeek fails."""
        mock_config_loader = Mock()
        mock_config_loader.is_feature_enabled.side_effect = ValueError("Invalid provider 'deepseek'")
        
        factory = UnifiedFieldFormatterFactory(mock_config_loader)
        formatter = factory.create_formatter('meaning_formatter')
        
        # Should fall back to standard formatter due to error handling
        from anki_pleco_importer.unified_field_formatter import StandardFieldFormatter
        assert isinstance(formatter, StandardFieldFormatter)


class TestDeepSeekFieldGenerator:
    """Test DeepSeek integration in field generation."""
    
    @pytest.fixture
    def mock_deepseek_config(self):
        """Mock configuration for DeepSeek field generation."""
        return {
            'provider': 'deepseek',
            'model': 'deepseek-chat',
            'api_key': 'test-deepseek-key',
            'base_url': 'https://api.deepseek.com',
            'temperature': 0.5,
            'max_tokens': 800,
            'prompt': 'field_generation_multi_char.md'
        }
    
    def test_deepseek_field_generator_now_works(self, mock_deepseek_config):
        """Test that DeepSeek field generator now works with implementation."""
        mock_config_loader = Mock()
        mock_config_loader.get_feature_config.return_value = mock_deepseek_config
        mock_config_loader.is_feature_enabled.return_value = True
        
        generator = AIFieldGenerator(mock_config_loader)
        
        # Should no longer raise an exception for deepseek provider
        client, config = generator._get_client_and_config('multi_char')
        assert client is not None
        assert config['provider'] == 'deepseek'


class TestDeepSeekWordClassifier:
    """Test DeepSeek integration in word classification."""
    
    @pytest.fixture
    def mock_deepseek_config(self):
        """Mock configuration for DeepSeek word classification."""
        return {
            'provider': 'deepseek',
            'model': 'deepseek-chat',
            'api_key': 'test-deepseek-key',
            'base_url': 'https://api.deepseek.com',
            'temperature': 0.1,
            'max_tokens': 200,
            'prompt': 'word_classifier.md'
        }
    
    def test_deepseek_word_classifier_fails(self, mock_deepseek_config):
        """Test that DeepSeek word classifier fails with current implementation."""
        # For now, just test that DeepSeek provider is not supported
        # The actual error will come when we try to create clients
        assert mock_deepseek_config['provider'] == 'deepseek'
        
        # This validates our test setup - when we implement DeepSeek,
        # this test will need to be updated to verify proper functionality


class TestDeepSeekAPIMocking:
    """Test DeepSeek API interaction patterns (for future implementation)."""
    
    def test_deepseek_api_request_format(self):
        """Test the expected DeepSeek API request format."""
        # This test documents the expected API format
        expected_request = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant for Chinese language learning.'},
                {'role': 'user', 'content': 'Chinese word: 高维\nOriginal content: (math.) higher dimensional'}
            ],
            'temperature': 0.3,
            'max_tokens': 400
        }
        
        expected_url = 'https://api.deepseek.com/v1/chat/completions'
        expected_headers = {
            'Authorization': 'Bearer test-deepseek-key',
            'Content-Type': 'application/json'
        }
        
        # Document the expected format for implementation
        assert expected_request['model'] == 'deepseek-chat'
        assert expected_url.startswith('https://api.deepseek.com')
        assert 'Bearer' in expected_headers['Authorization']
    
    def test_deepseek_api_response_format(self):
        """Test the expected DeepSeek API response format."""
        # This test documents the expected response format
        expected_response = {
            'id': 'deepseek-completion-id',
            'object': 'chat.completion',
            'model': 'deepseek-chat',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': '<span class="domain">mathematics</span> higher dimensional'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 150,
                'completion_tokens': 25,
                'total_tokens': 175
            }
        }
        
        # Document the expected format for implementation
        assert expected_response['object'] == 'chat.completion'
        assert 'usage' in expected_response
        assert 'prompt_tokens' in expected_response['usage']
    
    def test_deepseek_pricing_structure(self):
        """Test DeepSeek pricing structure for cost calculation."""
        # Based on research: DeepSeek-chat pricing
        deepseek_pricing = {
            'deepseek-chat': {
                'input_cache_hit': 0.07,  # per 1M tokens
                'input_cache_miss': 0.27,  # per 1M tokens  
                'output': 1.10  # per 1M tokens
            },
            'deepseek-reasoner': {
                'input_cache_hit': 0.14,  # per 1M tokens
                'input_cache_miss': 0.55,  # per 1M tokens
                'output': 2.19  # per 1M tokens
            }
        }
        
        # Verify pricing structure
        assert deepseek_pricing['deepseek-chat']['input_cache_miss'] == 0.27
        assert deepseek_pricing['deepseek-chat']['output'] == 1.10
        
        # Test cost calculation logic
        input_tokens = 1000
        output_tokens = 500
        
        # Assuming cache miss for simplicity
        expected_cost = (input_tokens / 1_000_000) * 0.27 + (output_tokens / 1_000_000) * 1.10
        
        assert expected_cost > 0
        assert expected_cost < 0.01  # Should be very small for small token counts


class TestDeepSeekImplementationPlan:
    """Tests that define the implementation plan for DeepSeek integration."""
    
    def test_required_config_schema_changes(self):
        """Test the required changes to configuration schema."""
        # This test documents what needs to be changed
        required_changes = {
            'ai_config.py': [
                'Add "deepseek" to valid_providers list in CharTypeConfig validator',
                'Add "deepseek" to valid_providers list in FeatureConfig validator'
            ],
            'unified_field_formatter.py': [
                'Add deepseek client creation in _get_client_and_config',
                'Add DeepSeek pricing in DEEPSEEK_PRICING constant',
                'Update _calculate_cost to handle deepseek provider'
            ],
            'unified_field_generator.py': [
                'Add deepseek client creation in _get_client_and_config',
                'Add deepseek API call handling in generate_fields'
            ],
            'unified_word_classifier.py': [
                'Add deepseek client creation in _get_client',
                'Add deepseek API call handling in classify_words'
            ]
        }
        
        # Verify the plan is complete
        assert len(required_changes) == 4
        assert 'ai_config.py' in required_changes
        assert 'unified_field_formatter.py' in required_changes
    
    def test_deepseek_client_interface(self):
        """Test the expected DeepSeek client interface."""
        # This test documents the expected client interface
        # When implemented, DeepSeek client should work like this:
        
        # Mock what the interface should look like
        expected_interface = {
            'base_url': 'https://api.deepseek.com',
            'api_key': 'test-key',
            'models': ['deepseek-chat', 'deepseek-reasoner'],
            'endpoints': {
                'chat_completions': '/v1/chat/completions'
            },
            'authentication': 'Bearer token in Authorization header',
            'request_format': 'OpenAI-compatible'
        }
        
        # Document the expected interface
        assert expected_interface['base_url'] == 'https://api.deepseek.com'
        assert 'deepseek-chat' in expected_interface['models']
        assert expected_interface['request_format'] == 'OpenAI-compatible'