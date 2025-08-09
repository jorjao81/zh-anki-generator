# Gemini Configuration

This directory contains configuration files for using Google Gemini 2.5 models to generate etymology and structural decomposition fields for Chinese characters.

## Files

- `prompt.md` – System prompt for the model describing the expected JSON structure.
- `config.gemini-2.5-flash-lite.json` – Most cost-effective configuration using `gemini-2.5-flash-lite` model.
- `config.gemini-2.5-flash.json` – Balanced configuration using `gemini-2.5-flash` model with better quality.
- `config.gemini-2.5-pro.json` – Highest quality configuration using `gemini-2.5-pro` model (most expensive).
- `examples/` – HTML snippets showing the expected formatting for the generated fields based on the card CSS.

## Model Comparison (per 1M tokens)

| Model | Input Cost | Output Cost | Use Case |
|-------|------------|-------------|----------|
| **gemini-2.5-flash-lite** | $0.10 | $0.40 | High-volume processing, cost optimization |
| **gemini-2.5-flash** | $0.30 | $2.50 | Balanced quality and cost |
| **gemini-2.5-pro** | $1.25 | $10.00 | Highest quality, research use |

## Usage

```bash
# Most cost-effective (recommended for large batches)
python -m anki_pleco_importer.cli convert input.tsv --use-gemini --gemini-config gemini/config.gemini-2.5-flash-lite.json

# Balanced quality and cost
python -m anki_pleco_importer.cli convert input.tsv --use-gemini --gemini-config gemini/config.gemini-2.5-flash.json

# Highest quality (expensive)
python -m anki_pleco_importer.cli convert input.tsv --use-gemini --gemini-config gemini/config.gemini-2.5-pro.json
```

## Setup

1. Get a Google AI Studio API key from https://aistudio.google.com/apikey
2. Set the environment variable: `export GEMINI_API_KEY="your-api-key"`
3. Or store it in 1Password and update `.envrc` as shown in the project root.

## Cost Comparison

For a typical batch of 1000 characters:
- **Gemini 2.5 Flash Lite**: ~$1.50 (most economical)
- **Gemini 2.5 Flash**: ~$7.50 
- **Gemini 2.5 Pro**: ~$30.00
- **GPT-5**: ~$120.00

Gemini 2.5 Flash Lite provides excellent value for money while maintaining good quality etymology and structural analysis.