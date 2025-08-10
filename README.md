# Anki Pleco Importer

A Python application that converts Pleco flashcard exports to Anki-compatible CSV format for Chinese language learning with AI-powered enhancements and custom card themes.

## ✨ Features

### Core Functionality
- **TSV to CSV Conversion**: Transform Pleco TSV exports into Anki-importable CSV format
- **Intelligent Definition Parsing**: Extract and format complex definitions with multiple meanings
- **Pinyin Conversion**: Convert numbered pinyin (e.g., `mi2shang4`) to toned pinyin (e.g., `míshàng`)
- **Semantic Decomposition**: Automatic character breakdown with meaning explanations using hanzipy
- **Example Extraction**: Intelligently separate Chinese examples from definitions
- **HTML Formatting**: Apply rich formatting for parts of speech and domain markers
- **Anki Export Parsing**: Parse existing Anki exports (Chinese.txt) for analysis and processing

### AI-Powered Enhancements
- **GPT Integration**: Generate etymology and structural analysis using OpenAI GPT models
- **Gemini Integration**: Alternative AI provider with Google's Gemini 2.5 models for cost-effective processing
- **Audio Generation**: Multi-provider audio generation (Forvo, Azure TTS) with caching
- **Usage Tracking**: Monitor AI costs and token usage across providers

### Custom Anki Templates
- **Phonetic Series Cards**: Specialized card type for learning Chinese phonetic components
- **10 Visual Themes**: Multiple design options from compact to creative layouts:
  - Default Match, Compact List, Timeline Horizontal, Radial Circle
  - Newspaper Columns, Zen Minimal, Vintage Paper, Neon Cyberpunk
  - Card Tiles, Compact List 2
- **Responsive Design**: All themes optimized for desktop and mobile viewing

## 🚀 Installation

### Requirements
- Python 3.8+

### Install from Source
```bash
git clone <repository-url>
cd anki-pleco-importer-python
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

## 📖 Usage

### Basic Conversion
```bash
anki-pleco-importer path/to/your/pleco_export.tsv
```

### AI-Enhanced Processing
```bash
# With GPT integration
anki-pleco-importer input.tsv --use-gpt --gpt-config gpt_config.json

# With Gemini integration (cost-effective)
anki-pleco-importer input.tsv --use-gemini --gemini-config gemini_config.json --gemini-model gemini-2.5-flash

# With audio generation
anki-pleco-importer input.tsv --audio --audio-providers "forvo,azure_tts"
```

### Anki Export Analysis
```bash
# Analyze existing Anki exports
anki-pleco-importer Chinese.txt --summary
```

### Configuration Files
Create JSON config files for AI providers:

**gpt_config.json**:
```json
{
    "api_key": "your-openai-key",
    "model": "gpt-4o-mini",
    "max_tokens": 500
}
```

**gemini_config.json**:
```json
{
    "api_key": "your-gemini-key", 
    "model": "gemini-2.5-flash"
}
```

### Example Input (TSV format)
```
迷上	mi2shang4	to become fascinated with; to become obsessed with
吟唱	yin2chang4	verb sing (a verse); chant
动弹	dong4tan5	verb move; stir 机器不动弹了。 Jīqì bù dòngtan le. The machine has stopped.
```

### Example Output (CSV format)
```
迷上,míshàng,,to become fascinated with; to become obsessed with,,,迷(mí - to bewilder/crazy about) + 上(shàng - on top/upon/above),,True,,True
吟唱,yínchàng,,<b>verb</b> sing (a verse); chant,,,吟(yín - to chant/to recite) + 唱(chàng - to sing/to call loudly),,True,,True
动弹,dòngtan,,<b>verb</b> move; stir,"机器不动弹了。 Jīqì bù dòngtan le. The machine has stopped.",,"动(dòng - to move/to set in movement) + 弹(dàn - bullet/shot/to spring)",,True,,True
```

## 🏗️ Architecture

### AnkiCard Model
The application uses a comprehensive `AnkiCard` model:

```python
@dataclass
class AnkiCard:
    pinyin: str                           # Toned pinyin (míshàng)
    simplified: str                       # Chinese characters (迷上)
    pronunciation: Optional[str] = None   # Audio file, you have to add this yourself
    meaning: str = ""                     # Formatted definition with HTML
    examples: Optional[List[str]] = None  # Chinese examples with translations
    phonetic_component: Optional[str] = None # used only for single characters
    structural_decomposition: Optional[str] = None  # Character decomposition
    similar_characters: Optional[List[str]] = None
    passive: bool = False                 # Default: True in conversion
    alternate_pronunciations: Optional[List[str]] = None
    nohearing: bool = False              # Default: True in conversion
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest

# BDD tests
behave

# All tests with coverage
pytest --cov=anki_pleco_importer
```

## 🛠️ Development

### Code Quality Tools
```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Lint code
flake8 src/ tests/
```

### Custom Anki Templates
Access the phonetic series templates in `anki-template/phonetic-series/`:

```bash
# Preview themes in browser
open anki-template/phonetic-series/themes/zen-minimal/preview.html
open anki-template/phonetic-series/themes/neon-cyberpunk/preview.html
# ... etc for all 10 themes
```

Available themes:
- **default-match**: Familiar blue/green gradient
- **compact-list**: Ultra space-efficient layout
- **timeline-horizontal**: Chronological timeline design
- **radial-circle**: Circular character arrangement
- **newspaper-columns**: Multi-column newspaper style
- **zen-minimal**: Clean, spacious design
- **vintage-paper**: Aged paper aesthetic
- **neon-cyberpunk**: Dark theme with neon colors
- **card-tiles**: Poker/casino card theme
- **compact-list-2**: Hybrid compact + timeline colors

### Project Structure
```
anki-pleco-importer-python/
├── src/anki_pleco_importer/
│   ├── cli.py              # Command-line interface
│   ├── pleco.py            # Pleco data parsing
│   ├── anki.py             # Anki card models  
│   ├── anki_parser.py      # Anki export parsing
│   ├── llm.py              # GPT/Gemini integration
│   ├── audio.py            # Audio generation
│   ├── chinese.py          # Chinese text processing
│   └── constants.py        # Configuration
├── anki-template/phonetic-series/  # Card themes
│   ├── themes/             # 10 visual themes
│   └── *.html             # Template files
├── features/               # BDD test scenarios
├── tests/                  # Unit tests
└── configs/               # AI provider configs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the existing patterns
4. Ensure all tests pass (`pytest && behave`)
5. Run code quality checks (`black`, `mypy`, `flake8`)
6. Submit a pull request

### Development Guidelines
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Use descriptive commit messages

## 📝 License

MIT License - see LICENSE file for details.

## 🔧 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed (`pip install -e ".[dev]"`)
- **Character Encoding**: Make sure TSV files are UTF-8 encoded
- **Empty Output**: Check that input TSV has correct format (Chinese\tpinyin\tdefinition)

### Debug Mode
Use `--help` to see all available options:
```bash
anki-pleco-importer --help
```
