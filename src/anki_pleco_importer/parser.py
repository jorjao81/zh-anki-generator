import pandas as pd
import re
from pathlib import Path
from typing import Union

from .pleco import PlecoEntry, PlecoCollection


class PlecoTSVParser:
    """Parser for Pleco TSV export files."""

    def _clean_chinese_field(self, chinese: str) -> str:
        """
        Clean the Chinese field by extracting only simplified characters,
        removing traditional characters in square brackets.
        
        Args:
            chinese: Raw Chinese field which may contain traditional characters in brackets
                    like "孤儿寡母[孤兒寡母]"
        
        Returns:
            Simplified Chinese characters only, e.g. "孤儿寡母"
        """
        if not chinese or not isinstance(chinese, str):
            return chinese
        
        # Extract simplified characters before square brackets
        # Pattern matches: simplified chars + optional [traditional chars]
        match = re.match(r"^([^\[]+)(\[.*\])?$", chinese.strip())
        if match:
            return match.group(1).strip()
        
        # If no brackets found, return as-is
        return chinese.strip()

    def parse_file(self, file_path: Union[str, Path]) -> PlecoCollection:
        """Parse a TSV file and return a PlecoCollection."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path, sep="\t", header=None, names=["chinese", "pinyin", "definition"])

        entries = []
        for _, row in df.iterrows():
            # Handle missing definition by using empty string
            definition = row["definition"] if pd.notna(row["definition"]) else ""
            # Clean Chinese field to extract only simplified characters
            cleaned_chinese = self._clean_chinese_field(row["chinese"])
            entry = PlecoEntry(chinese=cleaned_chinese, pinyin=row["pinyin"], definition=definition)
            entries.append(entry)

        return PlecoCollection(entries=entries)

    def parse_string(self, content: str) -> PlecoCollection:
        """Parse TSV content from a string and return a PlecoCollection."""
        from io import StringIO

        df = pd.read_csv(StringIO(content), sep="\t", header=None, names=["chinese", "pinyin", "definition"])

        entries = []
        for _, row in df.iterrows():
            # Handle missing definition by using empty string
            definition = row["definition"] if pd.notna(row["definition"]) else ""
            # Clean Chinese field to extract only simplified characters
            cleaned_chinese = self._clean_chinese_field(row["chinese"])
            entry = PlecoEntry(chinese=cleaned_chinese, pinyin=row["pinyin"], definition=definition)
            entries.append(entry)

        return PlecoCollection(entries=entries)
