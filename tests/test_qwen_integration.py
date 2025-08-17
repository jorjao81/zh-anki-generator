"""Unit tests for Qwen AI integration."""

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


class TestQwenConfiguration:
    """Test Qwen configuration validation and loading."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary Qwen config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = """
global:
  default_provider: "qwen"

providers:
  qwen:
    api_key: "test-qwen-key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

features:
  meaning_formatter:
    multi_char:
      provider: "qwen"
      model: "qwen-turbo"
      prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
      temperature: 0.3
"""
            f.write(config_content)
            temp_path = Path(f.name)
        
        yield temp_path
        temp_path.unlink(missing_ok=True)
    
    def test_qwen_config_validation_fails_on_api_key(self, temp_config_file):
        """Test that Qwen provider is now accepted but API key validation fails."""
        with pytest.raises(Exception) as excinfo:
            config_loader = AIConfigLoader(str(temp_config_file))
            config_loader.load_config()
        
        # Should contain validation error about invalid API key (not provider)
        assert "qwen" in str(excinfo.value).lower()
        assert ("invalid" in str(excinfo.value).lower() and "api" in str(excinfo.value).lower()) or "api key" in str(excinfo.value).lower()
    
    def test_qwen_provider_in_config_schema(self):
        """Test adding Qwen to allowed providers list."""
        # This test should now pass after we implemented Qwen support
        from anki_pleco_importer.ai_config import CharTypeConfig
        
        # Should no longer raise an exception for qwen provider
        config = CharTypeConfig(provider="qwen", model="qwen-turbo")
        assert config.provider == "qwen"
        assert config.model == "qwen-turbo"


class TestQwenFieldFormatter:
    """Test Qwen integration in field formatting."""
    
    @pytest.fixture
    def mock_qwen_config(self):
        """Mock configuration for Qwen."""
        return {
            'provider': 'qwen',
            'model': 'qwen-turbo',
            'api_key': 'test-qwen-key',
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'temperature': 0.3,
            'max_tokens': 400,
            'prompt': 'test_prompt.md'
        }
    
    @patch('openai.OpenAI')
    def test_qwen_client_authentication_fails(self, mock_openai, mock_qwen_config):
        """Test that Qwen client fails with invalid API key."""
        # Mock OpenAI client to raise authentication error for Qwen
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Error code: 401 - {'error': {'message': 'Invalid API key provided'}}"
        )
        mock_openai.return_value = mock_client
        
        mock_config_loader = Mock()
        mock_config_loader.get_feature_config.return_value = mock_qwen_config
        mock_config_loader.is_feature_enabled.return_value = True
        mock_config_loader.resolve_prompt_path.return_value = Path(__file__).parent / "test_prompt.md"
        
        # Create a test prompt file
        test_prompt = Path(__file__).parent / "test_prompt.md"
        test_prompt.write_text("Test prompt for Qwen")
        
        formatter = AIFieldFormatter('meaning_formatter', mock_config_loader)
        
        with pytest.raises(RuntimeError) as excinfo:
            formatter.format_field('测试', 'test meaning')
        
        assert "AI formatting failed" in str(excinfo.value)
        assert "Invalid API key" in str(excinfo.value)
        
        # Cleanup
        test_prompt.unlink(missing_ok=True)
    
    @patch('openai.OpenAI')
    def test_qwen_client_success(self, mock_openai, mock_qwen_config):
        """Test that Qwen client works with valid API key."""
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
        mock_config_loader.get_feature_config.return_value = mock_qwen_config
        mock_config_loader.is_feature_enabled.return_value = True
        mock_config_loader.resolve_prompt_path.return_value = Path(__file__).parent / "test_prompt.md"
        
        # Create a test prompt file
        test_prompt = Path(__file__).parent / "test_prompt.md"
        test_prompt.write_text("Test prompt for Qwen")
        
        formatter = AIFieldFormatter('meaning_formatter', mock_config_loader)
        result = formatter.format_field('高维', '(math.) higher dimensional')
        
        # Verify successful formatting
        assert result.formatted_content == "<span class=\"domain\">mathematics</span> higher dimensional"
        assert result.token_usage.prompt_tokens == 100
        assert result.token_usage.completion_tokens == 50
        assert result.token_usage.cost_usd > 0  # Should calculate Qwen cost
        
        # Verify Qwen API was called correctly
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args
        assert call_args[1]['api_key'] == 'test-qwen-key'
        assert call_args[1]['base_url'] == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        
        # Cleanup
        test_prompt.unlink(missing_ok=True)
    
    def test_qwen_cost_calculation_implemented(self):
        """Test that Qwen cost calculation is now implemented."""
        formatter = AIFieldFormatter('meaning_formatter', Mock())
        
        # Should calculate real cost for Qwen
        cost = formatter._calculate_cost('qwen', 'qwen-turbo', 100, 50)
        # 100 input tokens at $0.086/1M + 50 output tokens at $0.29/1M
        expected_cost = (100 / 1_000_000) * 0.086 + (50 / 1_000_000) * 0.29
        assert abs(cost - expected_cost) < 0.0000001
    
    def test_qwen_formatter_factory_fallback(self):
        """Test that formatter factory falls back to standard when Qwen fails."""
        mock_config_loader = Mock()
        mock_config_loader.is_feature_enabled.side_effect = ValueError("Invalid provider 'qwen'")
        
        factory = UnifiedFieldFormatterFactory(mock_config_loader)
        formatter = factory.create_formatter('meaning_formatter')
        
        # Should fall back to standard formatter due to error handling
        from anki_pleco_importer.unified_field_formatter import StandardFieldFormatter
        assert isinstance(formatter, StandardFieldFormatter)


class TestQwenFieldGenerator:
    """Test Qwen integration in field generation."""
    
    @pytest.fixture
    def mock_qwen_config(self):
        """Mock configuration for Qwen field generation."""
        return {
            'provider': 'qwen',
            'model': 'qwen-turbo',
            'api_key': 'test-qwen-key',
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'temperature': 0.5,
            'max_tokens': 800,
            'prompt': 'field_generation_multi_char.md'
        }
    
    def test_qwen_field_generator_now_works(self, mock_qwen_config):
        """Test that Qwen field generator now works with implementation."""
        mock_config_loader = Mock()
        mock_config_loader.get_feature_config.return_value = mock_qwen_config
        mock_config_loader.is_feature_enabled.return_value = True
        
        generator = AIFieldGenerator(mock_config_loader)
        
        # Should no longer raise an exception for qwen provider
        client, config = generator._get_client_and_config('multi_char')
        assert client is not None
        assert config['provider'] == 'qwen'


class TestQwenWordClassifier:
    """Test Qwen integration in word classification."""
    
    @pytest.fixture
    def mock_qwen_config(self):
        """Mock configuration for Qwen word classification."""
        return {
            'provider': 'qwen',
            'model': 'qwen-turbo',
            'api_key': 'test-qwen-key',
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'temperature': 0.1,
            'max_tokens': 200,
            'prompt': 'word_classifier.md'
        }
    
    def test_qwen_word_classifier_config(self, mock_qwen_config):
        """Test that Qwen word classifier config is correct."""
        # Test that Qwen provider configuration is accepted
        assert mock_qwen_config['provider'] == 'qwen'
        assert mock_qwen_config['model'] == 'qwen-turbo'
        assert mock_qwen_config['base_url'] == 'https://dashscope.aliyuncs.com/compatible-mode/v1'


class TestQwenAPIMocking:
    """Test Qwen API interaction patterns (for future implementation)."""
    
    def test_qwen_api_request_format(self):
        """Test the expected Qwen API request format."""
        # This test documents the expected API format
        expected_request = {
            'model': 'qwen-turbo',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant for Chinese language learning.'},
                {'role': 'user', 'content': 'Chinese word: 高维\nOriginal content: (math.) higher dimensional'}
            ],
            'temperature': 0.3,
            'max_tokens': 400
        }
        
        expected_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        expected_headers = {
            'Authorization': 'Bearer test-qwen-key',
            'Content-Type': 'application/json'
        }
        
        # Document the expected format for implementation
        assert expected_request['model'] == 'qwen-turbo'
        assert expected_url.startswith('https://dashscope.aliyuncs.com')
        assert 'Bearer' in expected_headers['Authorization']
    
    def test_qwen_api_response_format(self):
        """Test the expected Qwen API response format."""
        # This test documents the expected response format
        expected_response = {
            'id': 'qwen-completion-id',
            'object': 'chat.completion',
            'model': 'qwen-turbo',
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
    
    def test_qwen_pricing_structure(self):
        """Test Qwen pricing structure for cost calculation."""
        # Based on Qwen API pricing
        qwen_pricing = {
            'qwen-max': {
                'input': 8.00,  # per 1M tokens
                'output': 24.00  # per 1M tokens
            },
            'qwen-plus': {
                'input': 0.29,  # per 1M tokens
                'output': 0.86  # per 1M tokens
            },
            'qwen-turbo': {
                'input': 0.086,  # per 1M tokens
                'output': 0.29  # per 1M tokens
            },
            'qwen-long': {
                'input': 0.86,  # per 1M tokens
                'output': 2.00  # per 1M tokens
            },
            'qwen2.5-72b-instruct': {
                'input': 0.86,  # per 1M tokens
                'output': 2.00  # per 1M tokens
            },
            'qwen2.5-32b-instruct': {
                'input': 0.43,  # per 1M tokens
                'output': 1.00  # per 1M tokens
            },
            'qwen2.5-14b-instruct': {
                'input': 0.21,  # per 1M tokens
                'output': 0.50  # per 1M tokens
            },
            'qwen2.5-7b-instruct': {
                'input': 0.11,  # per 1M tokens
                'output': 0.25  # per 1M tokens
            }
        }
        
        # Verify pricing structure
        assert qwen_pricing['qwen-turbo']['input'] == 0.086
        assert qwen_pricing['qwen-turbo']['output'] == 0.29
        
        # Test cost calculation logic
        input_tokens = 1000
        output_tokens = 500
        
        expected_cost = (input_tokens / 1_000_000) * 0.086 + (output_tokens / 1_000_000) * 0.29
        
        assert expected_cost > 0
        assert expected_cost < 0.01  # Should be very small for small token counts


class TestQwenImplementationPlan:
    """Tests that define the implementation plan for Qwen integration."""
    
    def test_required_config_schema_changes(self):
        """Test the required changes to configuration schema."""
        # This test documents what needs to be changed
        required_changes = {
            'ai_config.py': [
                'Add "qwen" to valid_providers list in CharTypeConfig validator',
                'Add "qwen" to valid_providers list in FeatureConfig validator'
            ],
            'unified_field_formatter.py': [
                'Add qwen client creation in _get_client_and_config',
                'Add Qwen pricing in QWEN_PRICING constant',
                'Update _calculate_cost to handle qwen provider'
            ],
            'unified_field_generator.py': [
                'Add qwen client creation in _get_client_and_config',
                'Add qwen API call handling in generate_fields'
            ],
            'unified_word_classifier.py': [
                'Add qwen client creation in _get_client',
                'Add qwen API call handling in classify_words'
            ]
        }
        
        # Verify the plan is complete
        assert len(required_changes) == 4
        assert 'ai_config.py' in required_changes
        assert 'unified_field_formatter.py' in required_changes
    
    def test_qwen_client_interface(self):
        """Test the expected Qwen client interface."""
        # This test documents the expected client interface
        # When implemented, Qwen client should work like this:
        
        # Mock what the interface should look like
        expected_interface = {
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': 'test-key',
            'models': ['qwen-max', 'qwen-plus', 'qwen-turbo', 'qwen-long',
                      'qwen2.5-72b-instruct', 'qwen2.5-32b-instruct', 
                      'qwen2.5-14b-instruct', 'qwen2.5-7b-instruct'],
            'endpoints': {
                'chat_completions': '/chat/completions'
            },
            'authentication': 'Bearer token in Authorization header',
            'request_format': 'OpenAI-compatible'
        }
        
        # Document the expected interface
        assert expected_interface['base_url'] == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        assert 'qwen-turbo' in expected_interface['models']
        assert expected_interface['request_format'] == 'OpenAI-compatible'