# Phonetic Series Card Type

A specialized Anki card type for learning Chinese characters through their phonetic relationships and sound patterns.

## Overview

The Phonetic Series card type helps students understand how Chinese characters are connected through shared phonetic components. This approach is particularly effective for:

- Understanding character pronunciation patterns
- Building systematic vocabulary knowledge  
- Recognizing how semantic radicals combine with phonetic elements
- Learning multiple characters efficiently through pattern recognition

## Card Structure

### 1. **Phonetic Core**
- Main phonetic component (large display)
- Pronunciation with tone marker
- Core meaning

### 2. **Phonetic Family Grid**
- Related characters using the same phonetic component
- Each character shows:
  - Character, pinyin, and tone
  - English meaning
  - Radical + phonetic breakdown
  - Semantic analysis

### 3. **Phonetic Evolution**
- Historical pronunciation development
- Shows progression from ancient to modern sounds
- Helps understand pronunciation variations

### 4. **Memory Aid**
- Study tips and mnemonic devices
- Pattern recognition guidance
- Common meaning themes

### 5. **Series Statistics**
- Total characters in the phonetic series
- Number displayed on current card

## Design Features

- **Responsive layout** - Works on desktop and mobile
- **Color-coded elements** - Different colors for radicals, phonetics, and tones
- **Hover effects** - Interactive elements for better engagement
- **Clean typography** - Optimized for Chinese character display
- **Semantic HTML** - Proper structure for accessibility and maintenance

## Usage in Anki

### Front Template
```html
<div class="phonetic-core">
    <div class="core-character">
        <span class="hanzi-large">{{Phonetic-Component}}</span>
        <div class="core-pronunciation">
            <span class="pinyin">{{Core-Pinyin}}</span>
            <span class="tone-marker">{{Tone}}</span>
        </div>
    </div>
</div>
```

### Back Template
Include the full phonetic family grid and analysis sections.

### Styling
```html
<link rel="stylesheet" href="_phonetic-series-style.css">
```

## Recommended Fields

- `Phonetic-Component`: Main phonetic element (e.g., 青)
- `Core-Pinyin`: Pronunciation of core component
- `Core-Meaning`: Basic meaning of phonetic component
- `Family-Characters`: JSON/HTML of related characters
- `Evolution-Data`: Historical pronunciation information
- `Memory-Aid`: Study tips and mnemonics
- `Series-Count`: Total number of characters in series

## Educational Benefits

1. **Pattern Recognition**: Students learn to identify phonetic patterns
2. **Systematic Learning**: Related characters learned together
3. **Historical Context**: Understanding of language evolution
4. **Memory Enhancement**: Phonetic relationships aid recall
5. **Pronunciation Improvement**: Sound pattern awareness

## File Structure

```
phonetic-series/
├── phonetic-series.html     # Main card template
├── style.css                # Complete styling
├── preview.html             # Visual preview/demo
└── README.md               # This documentation
```

## Customization

The CSS uses custom properties (CSS variables) for easy theming:

```css
:root {
    --primary-blue: #2563eb;
    --light-blue: #dbeafe;
    --green-accent: #10b981;
    --orange-accent: #f59e0b;
    /* ... */
}
```

Change these values to customize colors and create variants for different phonetic series or difficulty levels.