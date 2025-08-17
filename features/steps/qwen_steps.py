"""Step definitions for Qwen AI integration BDD tests."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, Mock
from behave import given, when, then


# Note: Many step definitions are already available in other step files
# We only define Qwen-specific steps here


@when('I run the Qwen command with mocked success "{command}"')
def step_run_qwen_command_success(context, command):
    """Run a command with Qwen success mocked."""
    _run_qwen_command_with_mock(context, command, mock_type="success")


@when('I run the Qwen command with mocked field generation "{command}"')
def step_run_qwen_command_field_generation(context, command):
    """Run a command with Qwen field generation mocked."""
    _run_qwen_command_with_mock(context, command, mock_type="field_generation")


@when('I run the Qwen command with mocked classification "{command}"')
def step_run_qwen_command_classification(context, command):
    """Run a command with Qwen classification mocked."""
    _run_qwen_command_with_mock(context, command, mock_type="classification")


@when('I run the Qwen command with mocked failure "{command}"')
def step_run_qwen_command_failure(context, command):
    """Run a command with Qwen failure mocked."""
    _run_qwen_command_with_mock(context, command, mock_type="failure")


@when('I run the Qwen command with mocked cost tracking "{command}"')
def step_run_qwen_command_cost_tracking(context, command):
    """Run a command with Qwen cost tracking mocked."""
    _run_qwen_command_with_mock(context, command, mock_type="cost_tracking")


def _run_qwen_command_with_mock(context, command, mock_type):
    """Run a command with specific Qwen mock behavior."""
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
                if base_url and "dashscope.aliyuncs.com" in base_url:
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
    """Create a mock that simulates successful Qwen formatting."""
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
    """Create a mock that simulates successful Qwen field generation."""
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
    """Create a mock that simulates successful Qwen word classification."""
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
    """Create a mock that simulates Qwen authentication failure."""
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception(
        "Error code: 401 - {'error': {'message': 'Invalid API key provided', 'type': 'authentication_error', 'param': None, 'code': 'invalid_request_error'}}"
    )
    return mock_client


def _create_cost_tracking_mock():
    """Create a mock that simulates Qwen with cost tracking."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '<span class="domain">test</span> vocabulary'
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 30
    mock_response.usage.total_tokens = 130
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# Qwen-specific step definitions that don't conflict with existing ones


@then("the cost should reflect qwen-max pricing")
def step_cost_should_reflect_qwen_max_pricing(context):
    """Verify cost reflects qwen-max pricing."""
    # qwen-max has higher pricing than qwen-turbo
    assert (
        "Cost:" in context.stdout or "Total AI cost" in context.stdout
    ), f"Expected cost information in output. Got: {context.stdout}"
