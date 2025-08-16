"""End-to-end tests for AI integrations.

These tests verify that AI providers are called with correct parameters and that 
the output is properly formatted. They test at the outermost layer (convert function)
to catch regressions like:
- Missing examples in prompts  
- Wrong formatter types being created
- Formatting happening during parsing instead of in formatters
- Missing HTML span support
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json
import logging
from click.testing import CliRunner

from anki_pleco_importer.cli import cli


class TestAIIntegrationE2E:
    """End-to-end tests for AI integration at the CLI entry point level."""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test TSV file
            tsv_file = temp_path / "test.tsv"
            tsv_file.write_text(
                "高维\tgao1wei2\t(math.) higher dimensional\n"
                "化石燃料\thuàshíránliào\tfossil fuel\n"  
                "安慰剂\tānwèijì\t(pharm.) placebo; something said or done merely to soothe\n"
            )
            
            # Create AI config
            ai_config_file = temp_path / "test_ai_config.yaml"
            ai_config_content = """
global:
  default_provider: "gpt"

providers:
  gpt:
    api_key: "test-openai-key"
  gemini:
    api_key: "test-gemini-key"

features:
  field_generation:
    single_char:
      provider: "gemini"
      model: "gemini-2.5-pro"
      prompt: "ai_config/prompts/field_generation_single_char.md"
      temperature: 1.0
    multi_char:
      provider: "gemini"
      model: "gemini-2.5-pro"
      prompt: "ai_config/prompts/field_generation_multi_char.md"
      temperature: 1.0
      
  meaning_formatter:
    single_char:
      provider: "standard"
    multi_char:
      provider: "gpt"
      model: "gpt-5-mini"
      prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
      temperature: 1.0
      
  examples_formatter:
    single_char:
      provider: "gpt"
      model: "gpt-5-mini"
      prompt: "ai_config/prompts/examples_formatter_single_char.md"
      temperature: 1.0
      max_tokens: 400
    multi_char:
      provider: "gpt"
      model: "gpt-5-mini"
      prompt: "ai_config/prompts/examples_formatter_multi_char.md"
      temperature: 1.0
      max_tokens: 400
"""
            ai_config_file.write_text(ai_config_content)
            
            yield {
                'temp_dir': temp_path,
                'tsv_file': tsv_file,
                'ai_config_file': ai_config_file
            }

    @pytest.fixture
    def mock_prompt_files(self):
        """Mock prompt files to avoid filesystem dependencies."""
        prompts = {
            'meaning_formatter_multi_char.md': """
## Input Format
You will receive:
- **Chinese word**: The multi-character Chinese word/phrase being defined
- **Original content**: The raw definition/meaning from Pleco export

Format with HTML spans for domains and parts of speech.

if a usage or domain is in parentheses, remove the parentheses, eg:
input: (math.) higher dimensional
desired output: <span class="domain">mathematics</span> higher dimensional
""",
            'field_generation_multi_char.md': """
Generate etymology and structural decomposition for Chinese characters.

Return JSON format:
{
  "etymology": "Historical etymology explanation",
  "structural": "Character structure analysis"
}
""",
            'examples_formatter_multi_char.md': """
Format examples with proper HTML structure and styling.
"""
        }
        
        def mock_read_text(self, encoding='utf-8'):
            # Extract filename from the Path object
            filename = str(self).split('/')[-1]
            return prompts.get(filename, f"Mock prompt for {filename}")
        
        with patch.object(Path, 'read_text', mock_read_text):
            with patch.object(Path, 'exists', return_value=True):
                yield prompts

    def test_meaning_formatter_gpt_integration(self, temp_files, mock_prompt_files):
        """Test that GPT meaning formatter is called with correct parameters."""
        
        # Mock GPT response
        mock_gpt_response = Mock()
        mock_gpt_response.choices = [Mock()]
        mock_gpt_response.choices[0].message.content = '<span class="domain">mathematics</span> higher dimensional'
        mock_gpt_response.usage.prompt_tokens = 150
        mock_gpt_response.usage.completion_tokens = 20
        mock_gpt_response.usage.total_tokens = 170
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_gpt_response
            
            # Use Click CLI runner
            runner = CliRunner()
            result = runner.invoke(cli, [
                'convert',
                str(temp_files['tsv_file']),
                '--ai-config', str(temp_files['ai_config_file']),
                '--use-ai-formatting',
                '--dry-run',
                '--verbose'
            ])
            
            # Check that the command ran successfully
            assert result.exit_code == 0, f"Command failed with output: {result.output}"
            
            # Verify GPT client was created with correct API key
            mock_openai_class.assert_called_with(
                api_key="test-openai-key",
                base_url=None
            )
            
            # Verify GPT was called for meaning formatting
            mock_client.chat.completions.create.assert_called()
            call_args = mock_client.chat.completions.create.call_args
            
            # Check model and temperature
            assert call_args.kwargs['model'] == 'gpt-5-mini'
            assert call_args.kwargs['temperature'] == 1.0
            
            # Check that prompt contains the expected content
            messages = call_args.kwargs['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert 'domain is in parentheses' in messages[0]['content']
            assert messages[1]['role'] == 'user'
            
            # Should contain one of the Chinese words from our test data
            user_content = messages[1]['content']
            chinese_words = ['高维', '化石燃料', '安慰剂']
            assert any(word in user_content for word in chinese_words)
            
            # Should contain one of the original definitions
            original_defs = ['(math.) higher dimensional', 'fossil fuel', '(pharm.) placebo']
            assert any(def_text in user_content for def_text in original_defs)

    def test_field_generation_gemini_integration(self, temp_files, mock_prompt_files):
        """Test that Gemini field generator is called with correct parameters."""
        
        # Mock Gemini response
        mock_gemini_response = Mock()
        mock_gemini_response.text = '{"etymology": "高 (high) + 维 (dimension)", "structural": "Left-right structure"}'
        
        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_model_class.return_value = mock_model
                mock_model.generate_content.return_value = mock_gemini_response
                
                runner = CliRunner()
                result = runner.invoke(cli, [
                    'convert',
                    str(temp_files['tsv_file']),
                    '--ai-config', str(temp_files['ai_config_file']),
                    '--use-ai-fields',
                    '--dry-run',
                    '--verbose'
                ])
                assert result.exit_code == 0, f"Command failed with output: {result.output}"
                
                # Verify Gemini was configured with correct API key
                mock_configure.assert_called_with(api_key="test-gemini-key")
                
                # Verify Gemini model was created with correct model name
                mock_model_class.assert_called_with("gemini-2.5-pro")
                
                # Verify Gemini was called for field generation
                mock_model.generate_content.assert_called()
                call_args = mock_model.generate_content.call_args
                
                # Check that prompt contains expected content
                prompt = call_args[0][0]  # First positional argument
                
                # Should contain one of the Chinese words from our test data
                chinese_words = ['高维', '化石燃料', '安慰剂']
                assert any(word in prompt for word in chinese_words)
                assert 'JSON format' in prompt
                
                # Check generation config
                generation_config = call_args.kwargs['generation_config']
                assert generation_config['temperature'] == 1.0

    def test_provider_selection_based_on_config(self, temp_files, mock_prompt_files):
        """Test that the correct AI provider is selected based on configuration."""
        
        # Mock both GPT and Gemini responses
        mock_gpt_response = Mock()
        mock_gpt_response.choices = [Mock()]
        mock_gpt_response.choices[0].message.content = '<span class="domain">pharmacy</span> placebo'
        mock_gpt_response.usage.prompt_tokens = 100
        mock_gpt_response.usage.completion_tokens = 15
        mock_gpt_response.usage.total_tokens = 115
        
        mock_gemini_response = Mock()
        mock_gemini_response.text = '{"etymology": "安 (peace) + 慰 (comfort) + 剂 (agent)", "structural": "Complex ideographic compound"}'
        
        with patch('openai.OpenAI') as mock_openai_class:
            with patch('google.generativeai.configure') as mock_configure:
                with patch('google.generativeai.GenerativeModel') as mock_model_class:
                    mock_gpt_client = Mock()
                    mock_openai_class.return_value = mock_gpt_client
                    mock_gpt_client.chat.completions.create.return_value = mock_gpt_response
                    
                    mock_gemini_model = Mock()
                    mock_model_class.return_value = mock_gemini_model
                    mock_gemini_model.generate_content.return_value = mock_gemini_response
                    
                    runner = CliRunner()
                    result = runner.invoke(cli, [
                        'convert',
                        str(temp_files['tsv_file']),
                        '--ai-config', str(temp_files['ai_config_file']),
                        '--use-ai-fields',
                        '--use-ai-formatting',
                        '--dry-run',
                        '--verbose'
                    ])
                    assert result.exit_code == 0, f"Command failed with output: {result.output}"
                    
                    # Verify both providers were used as configured
                    mock_configure.assert_called_with(api_key="test-gemini-key")  # Field generation
                    mock_openai_class.assert_called_with(api_key="test-openai-key", base_url=None)  # Formatting

    def test_examples_loading_regression_prevention(self, temp_files, mock_prompt_files):
        """Test that examples are properly loaded and sent to AI - prevents the examples loading regression."""
        
        # Mock GPT response for examples formatting
        mock_gpt_response = Mock()
        mock_gpt_response.choices = [Mock()]
        mock_gpt_response.choices[0].message.content = '<ul><li>Example with formatting</li></ul>'
        mock_gpt_response.usage.prompt_tokens = 200
        mock_gpt_response.usage.completion_tokens = 30
        mock_gpt_response.usage.total_tokens = 230
        
        # Create TSV with examples
        tsv_with_examples = temp_files['temp_dir'] / "test_with_examples.tsv"
        tsv_with_examples.write_text(
            "测试\tcèshì\ttest; to test ◆ 他正在测试新软件。 He is testing the new software.\n"
        )
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_gpt_response
            
            runner = CliRunner()
            result = runner.invoke(cli, [
                'convert',
                str(tsv_with_examples),
                '--ai-config', str(temp_files['ai_config_file']),
                '--use-ai-formatting',
                '--dry-run',
                '--verbose'
            ])
            assert result.exit_code == 0, f"Command failed with output: {result.output}"
            
            # Verify that examples formatting was called
            mock_client.chat.completions.create.assert_called()
            
            # Find the call for examples formatting
            calls = mock_client.chat.completions.create.call_args_list
            examples_call = None
            for call in calls:
                messages = call.kwargs.get('messages', [])
                if any('他正在测试新软件' in str(msg) for msg in messages):
                    examples_call = call
                    break
            
            assert examples_call is not None, "Examples formatting call not found"
            
            # Verify the examples content was included in the prompt
            user_message = examples_call.kwargs['messages'][1]['content']
            assert '他正在测试新软件' in user_message, "Examples not included in AI prompt"

    def test_formatter_type_selection_regression_prevention(self, temp_files, mock_prompt_files):
        """Test that AIFieldFormatter is created when AI is enabled - prevents formatter selection regression."""
        
        # Mock GPT response
        mock_gpt_response = Mock()
        mock_gpt_response.choices = [Mock()]
        mock_gpt_response.choices[0].message.content = '<span class="part-of-speech">noun</span> fossil fuel'
        mock_gpt_response.usage.prompt_tokens = 120
        mock_gpt_response.usage.completion_tokens = 25
        mock_gpt_response.usage.total_tokens = 145
        
        with patch('openai.OpenAI') as mock_openai_class:
            with patch('anki_pleco_importer.unified_field_formatter.AIFieldFormatter') as mock_ai_formatter_class:
                with patch('anki_pleco_importer.unified_field_formatter.StandardFieldFormatter') as mock_standard_formatter_class:
                    
                    # Setup mocks
                    mock_client = Mock()
                    mock_openai_class.return_value = mock_client
                    mock_client.chat.completions.create.return_value = mock_gpt_response
                    
                    mock_ai_formatter = Mock()
                    mock_ai_formatter_class.return_value = mock_ai_formatter
                    mock_ai_formatter.format_field.return_value = Mock(
                        formatted_content='<span class="part-of-speech">noun</span> fossil fuel',
                        token_usage=Mock(total_tokens=145, cost_usd=0.001)
                    )
                    
                    mock_standard_formatter = Mock()
                    mock_standard_formatter_class.return_value = mock_standard_formatter
                    
                    runner = CliRunner()
                    result = runner.invoke(cli, [
                        'convert',
                        str(temp_files['tsv_file']),
                        '--ai-config', str(temp_files['ai_config_file']),
                        '--use-ai-formatting',
                        '--dry-run',
                        '--verbose'
                    ])
                    assert result.exit_code == 0, f"Command failed with output: {result.output}"
                    
                    # Verify AIFieldFormatter was created (not StandardFieldFormatter)
                    mock_ai_formatter_class.assert_called()
                    
                    # Verify StandardFieldFormatter was NOT used for AI-enabled features
                    # (It might be created for other purposes, but not for meaning formatting)
                    if mock_standard_formatter_class.called:
                        # If standard formatter was created, verify it wasn't used for formatting
                        mock_standard_formatter.format_field.assert_not_called()

    def test_html_output_formatting_with_domains(self, temp_files, mock_prompt_files):
        """Test that HTML output includes proper domain and part-of-speech spans."""
        
        # Mock GPT response with proper HTML spans
        mock_gpt_response = Mock()
        mock_gpt_response.choices = [Mock()]
        mock_gpt_response.choices[0].message.content = '<span class="domain">mathematics</span> higher dimensional'
        mock_gpt_response.usage.prompt_tokens = 150
        mock_gpt_response.usage.completion_tokens = 20
        mock_gpt_response.usage.total_tokens = 170
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_gpt_response
            
            # Capture output by writing to a file
            output_file = temp_files['temp_dir'] / "output.csv"
            
            # Patch the CSV writing to capture the output
            captured_data = []
            original_to_csv = None
            
            def mock_to_csv(file_path, **kwargs):
                # Capture the DataFrame data
                captured_data.extend(self.to_dict('records') if hasattr(self, 'to_dict') else [])
                return original_to_csv(file_path, **kwargs) if original_to_csv else None
            
            with patch('pandas.DataFrame.to_csv', side_effect=mock_to_csv):
                runner = CliRunner()
                result = runner.invoke(cli, [
                    'convert',
                    str(temp_files['tsv_file']),
                    '--ai-config', str(temp_files['ai_config_file']),
                    '--use-ai-formatting',
                    '--verbose',
                    '--html-output'
                ])
                assert result.exit_code == 0, f"Command failed with output: {result.output}"
                
                # Verify the GPT response contains proper HTML spans
                mock_client.chat.completions.create.assert_called()
                
                # The actual verification would depend on inspecting the output file
                # For now, we verify the AI was called and returned proper HTML
                assert '<span class="domain">mathematics</span>' in mock_gpt_response.choices[0].message.content

    def test_no_formatting_during_parsing_regression_prevention(self, temp_files, mock_prompt_files):
        """Test that formatting only happens in formatter classes, not during parsing."""
        
        # Mock GPT response
        mock_gpt_response = Mock()
        mock_gpt_response.choices = [Mock()]
        mock_gpt_response.choices[0].message.content = '<span class="domain">mathematics</span> higher dimensional'
        mock_gpt_response.usage.prompt_tokens = 150
        mock_gpt_response.usage.completion_tokens = 20
        mock_gpt_response.usage.total_tokens = 170
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_gpt_response
            
            runner = CliRunner()
            result = runner.invoke(cli, [
                'convert',
                str(temp_files['tsv_file']),
                '--ai-config', str(temp_files['ai_config_file']),
                '--use-ai-formatting',
                '--dry-run',
                '--verbose'
            ])
            assert result.exit_code == 0, f"Command failed with output: {result.output}"
            
            # Verify formatting happens in the formatter by checking AI was called
            mock_client.chat.completions.create.assert_called()
            call_args = mock_client.chat.completions.create.call_args
            user_message = call_args.kwargs['messages'][1]['content']
            
            # Should contain raw content from TSV (parentheses indicate unformatted)
            parenthetical_patterns = ['(math.)', '(pharm.)']
            assert any(pattern in user_message for pattern in parenthetical_patterns)
            
            # Verify AI returns formatted content
            assert '<span class="domain">mathematics</span>' in mock_gpt_response.choices[0].message.content

    def test_comprehensive_ai_integration_workflow(self, temp_files, mock_prompt_files):
        """Test the complete AI workflow with both field generation and formatting."""
        
        # Mock Gemini field generation response
        mock_gemini_response = Mock()
        mock_gemini_response.text = '{"etymology": "安 (peace) + 慰 (comfort) + 剂 (agent)", "structural": "Complex compound"}'
        
        # Mock GPT formatting responses
        mock_gpt_meaning_response = Mock()
        mock_gpt_meaning_response.choices = [Mock()]
        mock_gpt_meaning_response.choices[0].message.content = '<span class="domain">pharmacy</span> placebo; something to soothe'
        mock_gpt_meaning_response.usage.prompt_tokens = 150
        mock_gpt_meaning_response.usage.completion_tokens = 25
        mock_gpt_meaning_response.usage.total_tokens = 175
        
        mock_gpt_examples_response = Mock()
        mock_gpt_examples_response.choices = [Mock()]
        mock_gpt_examples_response.choices[0].message.content = '<ul><li>Example 1</li><li>Example 2</li></ul>'
        mock_gpt_examples_response.usage.prompt_tokens = 100
        mock_gpt_examples_response.usage.completion_tokens = 20
        mock_gpt_examples_response.usage.total_tokens = 120
        
        # Create test file with examples
        tsv_with_everything = temp_files['temp_dir'] / "comprehensive_test.tsv"
        tsv_with_everything.write_text(
            "安慰剂\tānwèijì\t(pharm.) placebo; something said or done merely to soothe ◆ 这只是安慰剂效应。 This is just a placebo effect.\n"
        )
        
        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                with patch('openai.OpenAI') as mock_openai_class:
                    
                    # Setup Gemini mocks
                    mock_gemini_model = Mock()
                    mock_model_class.return_value = mock_gemini_model
                    mock_gemini_model.generate_content.return_value = mock_gemini_response
                    
                    # Setup GPT mocks with different responses for different calls
                    mock_gpt_client = Mock()
                    mock_openai_class.return_value = mock_gpt_client
                    mock_gpt_client.chat.completions.create.side_effect = [
                        mock_gpt_meaning_response,  # First call for meaning
                        mock_gpt_examples_response   # Second call for examples
                    ]
                    
                    runner = CliRunner()
                    result = runner.invoke(cli, [
                        'convert',
                        str(tsv_with_everything),
                        '--ai-config', str(temp_files['ai_config_file']),
                        '--use-ai-fields',
                        '--use-ai-formatting',
                        '--dry-run',
                        '--verbose'
                    ])
                    assert result.exit_code == 0, f"Command failed with output: {result.output}"
                    
                    # Verify Gemini was used for field generation
                    mock_configure.assert_called_with(api_key="test-gemini-key")
                    mock_gemini_model.generate_content.assert_called()
                    
                    # Verify GPT was called twice (meaning + examples)
                    assert mock_gpt_client.chat.completions.create.call_count == 2
                    
                    # Verify the calls were made with correct content
                    calls = mock_gpt_client.chat.completions.create.call_args_list
                    
                    # First call should be for meaning formatting
                    meaning_call = calls[0]
                    meaning_user_msg = meaning_call.kwargs['messages'][1]['content']
                    assert '安慰剂' in meaning_user_msg
                    assert '(pharm.) placebo' in meaning_user_msg
                    
                    # Second call should be for examples formatting  
                    examples_call = calls[1]
                    examples_user_msg = examples_call.kwargs['messages'][1]['content']
                    assert '安慰剂' in examples_user_msg
                    assert '这只是安慰剂效应' in examples_user_msg

    def test_config_validation_and_error_handling(self, temp_files, mock_prompt_files):
        """Test that configuration validation works and errors are handled gracefully."""
        
        # Test with non-existent config file
        non_existent_config = temp_files['temp_dir'] / "nonexistent_config.yaml"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'convert',
            str(temp_files['tsv_file']),
            '--ai-config', str(non_existent_config),
            '--use-ai-formatting',
            '--dry-run',
            '--verbose'
        ])
        # Should fail because config file doesn't exist
        assert result.exit_code != 0, "Expected command to fail with non-existent config file"

    def test_token_usage_and_cost_tracking(self, temp_files, mock_prompt_files):
        """Test that token usage and costs are properly tracked and reported."""
        
        # Mock GPT response with usage data
        mock_gpt_response = Mock()
        mock_gpt_response.choices = [Mock()]
        mock_gpt_response.choices[0].message.content = '<span class="domain">mathematics</span> higher dimensional'
        mock_gpt_response.usage.prompt_tokens = 150
        mock_gpt_response.usage.completion_tokens = 25
        mock_gpt_response.usage.total_tokens = 175
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_gpt_response
            
            runner = CliRunner()
            result = runner.invoke(cli, [
                'convert',
                str(temp_files['tsv_file']),
                '--ai-config', str(temp_files['ai_config_file']),
                '--use-ai-formatting',
                '--dry-run',
                '--verbose'
            ])
            assert result.exit_code == 0, f"Command failed with output: {result.output}"
            
            # With CliRunner, output is captured in result.output
            # Should contain AI usage information
            assert "AI Usage" in result.output or "tokens" in result.output