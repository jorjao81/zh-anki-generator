"""Step definitions for Chinese 2 note type parsing tests."""

import tempfile
import os
from pathlib import Path
from behave import given, when, then
from src.anki_pleco_importer.anki_parser import AnkiExportParser
from src.anki_pleco_importer.cli import main
import subprocess
import sys


@given("the AnkiExportParser is available")
def step_anki_parser_available(context):
    """Initialize the AnkiExportParser."""
    context.parser = AnkiExportParser()


@given('I have an Anki export file with Chinese 2 note type data')
def step_create_chinese2_file(context):
    """Create a temporary file with Chinese 2 note type data."""
    # Create temporary file
    fd, context.test_file = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        # Write the content from the step
        content = context.text.strip()
        f.write(content)


@given('I have an Anki export file with mixed note type data')
def step_create_mixed_file(context):
    """Create a temporary file with mixed note type data."""
    # Create temporary file
    fd, context.test_file = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        # Write the content from the step
        content = context.text.strip()
        f.write(content)


@given('I have an Anki export file "{filename}" with Chinese 2 note type data')
def step_create_named_file(context, filename):
    """Create a file with the specified name."""
    context.test_file = os.path.join(tempfile.gettempdir(), filename)
    with open(context.test_file, 'w', encoding='utf-8') as f:
        content = context.text.strip()
        f.write(content)


@when("I parse the file")
def step_parse_file(context):
    """Parse the test file."""
    context.cards = context.parser.parse_file(Path(context.test_file))


@when('I run the summary command on "{filename}"')
def step_run_summary_command(context, filename):
    """Run the CLI summary command."""
    test_file = os.path.join(tempfile.gettempdir(), filename)
    
    # Run the CLI command
    try:
        result = subprocess.run([
            sys.executable, '-m', 'anki_pleco_importer.cli', 
            'summary', test_file
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        context.summary_output = result.stdout
        context.summary_stderr = result.stderr
        context.summary_returncode = result.returncode
        
    except Exception as e:
        context.summary_error = str(e)


@then("I should get {count:d} card")
@then("I should get {count:d} cards")
def step_check_card_count(context, count):
    """Check the number of cards parsed."""
    assert len(context.cards) == count, f"Expected {count} cards, got {len(context.cards)}"


@then("the card should have")
def step_check_card_fields(context):
    """Check card field values."""
    card = context.cards[0]
    for row in context.table:
        field_name = row['field']
        expected_value = row['value']
        
        if field_name == 'notetype':
            actual_value = card.notetype
        elif field_name == 'characters':
            actual_value = card.characters
        elif field_name == 'pinyin':
            actual_value = card.pinyin
        elif field_name == 'audio':
            actual_value = card.audio
        elif field_name == 'definitions':
            actual_value = card.definitions
        else:
            raise ValueError(f"Unknown field: {field_name}")
            
        assert actual_value == expected_value, f"Field {field_name}: expected '{expected_value}', got '{actual_value}'"


@then('the clean characters should be "{expected_chars}"')
def step_check_clean_characters(context, expected_chars):
    """Check the clean characters extraction."""
    card = context.cards[0]
    actual_chars = card.get_clean_characters()
    assert actual_chars == expected_chars, f"Expected clean characters '{expected_chars}', got '{actual_chars}'"


@then('card {card_num:d} should have notetype "{expected_notetype}" and characters "{expected_chars}"')
def step_check_specific_card(context, card_num, expected_notetype, expected_chars):
    """Check specific card properties."""
    card = context.cards[card_num - 1]  # Convert to 0-based index
    assert card.notetype == expected_notetype, f"Card {card_num} notetype: expected '{expected_notetype}', got '{card.notetype}'"
    assert card.characters == expected_chars, f"Card {card_num} characters: expected '{expected_chars}', got '{card.characters}'"


@then("the summary should show {count:d} total cards")
def step_check_summary_total(context, count):
    """Check total card count in summary."""
    assert f"Total cards: {count}" in context.summary_output, f"Expected 'Total cards: {count}' in summary output: {context.summary_output}"


@then("the summary should show {count:d} unique characters")
def step_check_summary_unique_characters(context, count):
    """Check unique character count in summary."""
    assert f"Total unique characters: {count}" in context.summary_output, f"Expected 'Total unique characters: {count}' in summary output: {context.summary_output}"


def after_scenario(context, scenario):
    """Clean up temporary files after each scenario."""
    if hasattr(context, 'test_file') and os.path.exists(context.test_file):
        os.unlink(context.test_file)