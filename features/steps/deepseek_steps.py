"""Step definitions for DeepSeek AI integration BDD tests."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, Mock
from behave import given, when, then


@given("I have a test TSV file with Chinese words")
def step_prepare_test_environment(context):
    """Prepare the test environment for DeepSeek integration tests."""
    # context.test_files_dir is already set up in environment.py
    pass


@given("I have a DeepSeek AI configuration file")
def step_prepare_deepseek_config(context):
    """Prepare for DeepSeek configuration."""
    pass  # Implementation will be in specific config steps


@given('I have the sample TSV file "{filename}" with content')
def step_create_tsv_file(context, filename):
    """Create a TSV file with the given content."""
    file_path = context.test_files_dir / filename
    file_path.write_text(context.text.strip(), encoding="utf-8")


@given('I have a DeepSeek AI config file "{filename}" with content')
def step_create_deepseek_config_file(context, filename):
    """Create a DeepSeek AI configuration file."""
    file_path = context.test_files_dir / filename
    file_path.write_text(context.text.strip(), encoding="utf-8")


@given('I have a sample EPUB file "{filename}"')
def step_create_sample_epub(context, filename):
    """Create a minimal sample EPUB file for testing."""
    file_path = context.test_files_dir / filename
    # Create a minimal EPUB structure
    epub_content = b"PK\x03\x04\x14\x00\x00\x00\x08\x00"  # Minimal ZIP header
    file_path.write_bytes(epub_content)


@given('I have an Anki export file "{filename}" with basic vocabulary')
def step_create_anki_export(context, filename):
    """Create a basic Anki export file."""
    file_path = context.test_files_dir / filename
    content = """你好	nǐhǎo	hello
谢谢	xièxie	thank you
再见	zàijiàn	goodbye"""
    file_path.write_text(content, encoding="utf-8")


@then("the command should fail with invalid API key")
def step_command_should_fail_invalid_api_key(context):
    """Verify the command failed because of invalid API key."""
    assert (
        context.exit_code != 0
    ), f"Expected command to fail with invalid API key, but it succeeded. Stdout: {context.stdout}. Stderr: {context.stderr}"


# Note: Many step definitions are already available in other step files
# We only define DeepSeek-specific steps here


@when('I run the DeepSeek command with mocked success "{command}"')
def step_run_deepseek_command_success(context, command):
    """Run a command with DeepSeek success mocked."""
    _run_deepseek_command_with_mock(context, command, mock_type="success")


@when('I run the DeepSeek command with mocked field generation "{command}"')
def step_run_deepseek_command_field_generation(context, command):
    """Run a command with DeepSeek field generation mocked."""
    _run_deepseek_command_with_mock(context, command, mock_type="field_generation")


@when('I run the DeepSeek command with mocked classification "{command}"')
def step_run_deepseek_command_classification(context, command):
    """Run a command with DeepSeek classification mocked."""
    _run_deepseek_command_with_mock(context, command, mock_type="classification")


@when('I run the DeepSeek command with mocked failure "{command}"')
def step_run_deepseek_command_failure(context, command):
    """Run a command with DeepSeek failure mocked."""
    _run_deepseek_command_with_mock(context, command, mock_type="failure")


@when('I run the DeepSeek command with mocked cost tracking "{command}"')
def step_run_deepseek_command_cost_tracking(context, command):
    """Run a command with DeepSeek cost tracking mocked."""
    _run_deepseek_command_with_mock(context, command, mock_type="cost_tracking")


def _run_deepseek_command_with_mock(context, command, mock_type):
    """Run a command with specific DeepSeek mock behavior."""
    try:
        with patch("openai.OpenAI") as mock_openai:
            if mock_type == "success":
                mock_client = _create_success_mock()
            elif mock_type == "field_generation":
                mock_client = _create_field_generation_mock()
            elif mock_type == "classification":
                mock_client = _create_classification_mock()
            elif mock_type == "failure":
                mock_client = _create_failure_mock()
            elif mock_type == "cost_tracking":
                mock_client = _create_cost_tracking_mock()
            else:
                mock_client = _create_success_mock()

            def mock_openai_factory(api_key=None, base_url=None):
                if base_url and "deepseek" in base_url:
                    return mock_client
                # Return working mock for other providers
                working_client = Mock()
                working_response = Mock()
                working_response.choices = [Mock()]
                working_response.choices[0].message.content = "Standard response"
                working_response.usage.prompt_tokens = 10
                working_response.usage.completion_tokens = 5
                working_response.usage.total_tokens = 15
                working_client.chat.completions.create.return_value = working_response
                return working_client

            mock_openai.side_effect = mock_openai_factory

            # Execute command
            if command.startswith("anki-pleco-importer"):
                cmd_parts = command.split()
                cmd_parts[0] = sys.executable
                cmd_parts.insert(1, "-m")
                cmd_parts.insert(2, "anki_pleco_importer.cli")

                # Replace relative file paths with absolute paths
                for i, part in enumerate(cmd_parts):
                    if (
                        part.endswith(".tsv")
                        or part.endswith(".yaml")
                        or part.endswith(".epub")
                        or part.endswith(".txt")
                    ):
                        cmd_parts[i] = str(context.test_files_dir / part)

                result = subprocess.run(
                    cmd_parts,
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
            else:
                result = subprocess.run(
                    command.split(),
                    cwd=context.test_files_dir,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )

            context.command_result = result
            context.exit_code = result.returncode
            context.stdout = result.stdout
            context.stderr = result.stderr

    except Exception as e:
        context.command_error = str(e)
        context.exit_code = 1


def _create_success_mock():
    """Create a mock that simulates successful DeepSeek formatting."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '<span class="domain">mathematics</span> higher dimensional'
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 25
    mock_response.usage.total_tokens = 175
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _create_field_generation_mock():
    """Create a mock that simulates successful DeepSeek field generation."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = """
    {
        "etymology_html": "<p>化石 (fossil) + 燃料 (fuel)</p>",
        "structure_html": "<p>Compound word structure</p>"
    }
    """
    mock_response.usage.prompt_tokens = 200
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 250
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _create_classification_mock():
    """Create a mock that simulates successful DeepSeek word classification."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = """
    {
        "definition": "test vocabulary",
        "classification": "worth_learning"
    }
    """
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 120
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _create_failure_mock():
    """Create a mock that simulates DeepSeek authentication failure."""
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception(
        "Error code: 401 - {'error': {'message': 'Authentication Fails, Your api key: ****rmat is invalid', 'type': 'authentication_error', 'param': None, 'code': 'invalid_request_error'}}"
    )
    return mock_client


def _create_cost_tracking_mock():
    """Create a mock that simulates DeepSeek with cost tracking."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '<span class="domain">test</span> vocabulary'
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 30
    mock_response.usage.total_tokens = 130
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@then("the command should succeed")
def step_command_should_succeed(context):
    """Verify the command succeeded."""
    assert (
        context.exit_code == 0
    ), f"Expected command to succeed, but it failed. Exit code: {context.exit_code}. Stdout: {context.stdout}. Stderr: {context.stderr}"


@then("the output should show formatted meanings with domain markup")
def step_output_should_show_formatted_meanings(context):
    """Verify the output contains properly formatted meanings."""
    assert (
        '<span class="domain">mathematics</span>' in context.stdout
    ), f"Expected domain markup in output. Got: {context.stdout}"


@then("the cost should be calculated correctly for {num_words:d} words")
def step_cost_should_be_calculated_correctly(context, num_words):
    """Verify cost calculation for specified number of words."""
    # Check that cost calculation output is present
    assert (
        "Total AI cost" in context.stdout or "Cost:" in context.stdout
    ), f"Expected cost information in output. Got: {context.stdout}"

    # Check that it mentions the correct number of words
    assert (
        f"{num_words} word" in context.stdout
    ), f"Expected mention of {num_words} words in output. Got: {context.stdout}"
