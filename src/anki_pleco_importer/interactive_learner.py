"""Interactive learning interface for EPUB analysis."""

from __future__ import annotations

import click
import json
import sys
import tty
import termios
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass

from .unified_word_classifier import WordClassification


def get_single_key() -> str:
    """Get a single key press without requiring Enter."""
    try:
        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        # Set terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        
        # Read single character
        ch = sys.stdin.read(1)
        
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        return ch.lower()
    except:
        # Fallback to regular input if terminal handling fails
        return input().lower().strip()


@dataclass
class LearningSession:
    """Tracks a learning session state."""
    proper_names: Set[str] = None
    words_to_learn: List[Dict[str, str]] = None
    known_words: Set[str] = None
    
    def __post_init__(self):
        if self.proper_names is None:
            self.proper_names = set()
        if self.words_to_learn is None:
            self.words_to_learn = []
        if self.known_words is None:
            self.known_words = set()


class InteractiveLearner:
    """Interactive learning interface for managing vocabulary."""
    
    def __init__(self, names_file: Optional[Path] = None, known_words_file: Optional[Path] = None):
        """Initialize with file paths."""
        self.names_file = names_file or Path("names.txt")
        self.known_words_file = known_words_file or Path("known_words.txt")
        
        # Load existing files
        self.existing_names = self._load_file_set(self.names_file)
        self.existing_known_words = self._load_file_set(self.known_words_file)
    
    def _load_file_set(self, file_path: Path) -> Set[str]:
        """Load a set of words from a text file."""
        if not file_path.exists():
            return set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f if line.strip()}
        except Exception:
            return set()
    
    def _save_file_set(self, file_path: Path, words: Set[str]) -> None:
        """Save a set of words to a text file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                for word in sorted(words):
                    f.write(f"{word}\n")
        except Exception as e:
            click.echo(f"Warning: Failed to save {file_path}: {e}")
    
    def _save_pleco_file(self, words: List[Dict[str, str]], file_path: Path) -> None:
        """Save words in Pleco-compatible TSV format."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("chinese\tpinyin\tdefinition\n")
                for word_data in words:
                    f.write(f"{word_data['chinese']}\t{word_data.get('pinyin', '')}\t{word_data['definition']}\n")
        except Exception as e:
            click.echo(f"Warning: Failed to save Pleco file {file_path}: {e}")
    
    def process_classified_words(self, classifications: List[WordClassification]) -> LearningSession:
        """Process classified words interactively."""
        session = LearningSession()
        
        # Automatically handle proper names
        proper_names = [c for c in classifications if c.classification == 'proper_name']
        if proper_names:
            click.echo(f"\n🏷️  Found {len(proper_names)} proper names - automatically adding to names.txt:")
            for classification in proper_names:
                click.echo(f"  + {classification.word} ({classification.definition})")
                session.proper_names.add(classification.word)
        
        # Interactive processing for other words
        learning_candidates = [
            c for c in classifications 
            if c.classification in ['worth_learning', 'compositional']
        ]
        
        if not learning_candidates:
            click.echo("\nℹ️  No learning candidates found.")
            return session
        
        click.echo(f"\n📚 Found {len(learning_candidates)} potential learning words:")
        click.echo("\033[90mFor each word, press: \033[92m[y]\033[90mes to learn, \033[91m[n]\033[90mo to skip, \033[93m[q]\033[90muit\033[0m")
        click.echo("\033[90m" + "─" * 80 + "\033[0m")
        
        for i, classification in enumerate(learning_candidates, 1):
            # Show word information with ANSI formatting
            classification_emoji = "🟢" if classification.classification == "worth_learning" else "🟡"
            category_color = "\033[92m" if classification.classification == "worth_learning" else "\033[93m"
            category = "Worth Learning" if classification.classification == "worth_learning" else "Compositional"
            
            # Clear formatting and display word info
            click.echo(f"\n{classification_emoji} \033[90mWord {i}/{len(learning_candidates)}:\033[0m")
            click.echo(f"   \033[1;96m{classification.word}\033[0m")  # Bright cyan hanzi
            click.echo(f"   \033[94m{classification.definition}\033[0m")  # Blue definition  
            click.echo(f"   {category_color}{category}\033[0m")  # Green/yellow category
            
            while True:
                click.echo("\n\033[90mLearn this word? \033[92m[y]\033[90m/\033[91m[n]\033[90m/\033[93m[q]\033[90m: \033[0m", nl=False)
                choice = get_single_key()
                
                # Echo the choice for feedback
                if choice in ['y', 'n', 'q']:
                    click.echo(choice)
                else:
                    click.echo(f"\n\033[91mInvalid choice '{choice}'. Please press y, n, or q.\033[0m")
                    continue
                
                if choice == 'q':
                    click.echo("\033[93mQuitting interactive session...\033[0m")
                    return session
                elif choice == 'y':
                    session.words_to_learn.append({
                        'chinese': classification.word,
                        'pinyin': '',  # Will be filled by user or left empty
                        'definition': classification.definition
                    })
                    click.echo(f"\033[92m✅ Added '{classification.word}' to learning list\033[0m")
                    break
                elif choice == 'n':
                    session.known_words.add(classification.word)
                    click.echo(f"\033[90m➖ Added '{classification.word}' to known words\033[0m")
                    break
        
        return session
    
    def save_session(self, session: LearningSession, output_dir: Path = None) -> None:
        """Save learning session results to files."""
        if output_dir is None:
            output_dir = Path.cwd()
        
        # Update and save proper names
        if session.proper_names:
            all_names = self.existing_names | session.proper_names
            self._save_file_set(self.names_file, all_names)
            click.echo(f"\n📝 Updated {self.names_file} with {len(session.proper_names)} new names")
        
        # Update and save known words  
        if session.known_words:
            all_known = self.existing_known_words | session.known_words
            self._save_file_set(self.known_words_file, all_known)
            click.echo(f"📝 Updated {self.known_words_file} with {len(session.known_words)} new words")
        
        # Save learning words in Pleco format
        if session.words_to_learn:
            pleco_file = output_dir / "words_to_learn.tsv"
            self._save_pleco_file(session.words_to_learn, pleco_file)
            click.echo(f"📚 Created {pleco_file} with {len(session.words_to_learn)} words for Pleco import")
            click.echo("   Import this file into Pleco to add these words to your flashcards")
        
        # Summary
        total_processed = len(session.proper_names) + len(session.known_words) + len(session.words_to_learn)
        if total_processed > 0:
            click.echo(f"\n✅ Session complete! Processed {total_processed} words:")
            if session.proper_names:
                click.echo(f"   • {len(session.proper_names)} proper names → names.txt")
            if session.words_to_learn:
                click.echo(f"   • {len(session.words_to_learn)} learning words → words_to_learn.tsv")
            if session.known_words:
                click.echo(f"   • {len(session.known_words)} known words → known_words.txt")
        else:
            click.echo("\nℹ️  No changes made.")


def create_interactive_learner(names_file: Optional[Path] = None, known_words_file: Optional[Path] = None) -> InteractiveLearner:
    """Factory function to create an interactive learner."""
    return InteractiveLearner(names_file, known_words_file)