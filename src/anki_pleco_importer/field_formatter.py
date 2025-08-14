"""AI-powered field formatting for Anki cards."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel

from .llm import TokenUsage, FieldGenerator


class FieldFormattingResult(BaseModel):
    """Result from field formatting operation."""
    
    formatted_content: str
    token_usage: Optional[TokenUsage] = None


class FieldFormatter(ABC):
    """Interface for AI-powered field formatting."""
    
    @abstractmethod
    def format_field(self, hanzi: str, original_content: str) -> FieldFormattingResult:
        """Format a field using AI with the original content and hanzi as context."""
        raise NotImplementedError


class GptFieldFormatter(FieldFormatter):
    """Format fields using OpenAI GPT models."""
    
    # GPT pricing per 1M tokens (same as main LLM module)
    PRICING = {
        "gpt-5": {"input": 1.25, "output": 10.00, "cached_input": 0.125},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    }

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        prompt_path: Optional[str] = None,
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = 500,
    ) -> None:
        from openai import OpenAI
        from pathlib import Path

        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load formatting prompts for both single and multi-character
        self.single_char_prompt = ""
        self.multi_char_prompt = ""
        
        if prompt_path:
            prompt_path_obj = Path(prompt_path)
            
            # Load single character prompt (the default one)
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.single_char_prompt = f.read()

            # Load multi-character prompt
            single_char_dir = prompt_path_obj.parent.name
            if single_char_dir.endswith("_single_char"):
                multi_char_dir = single_char_dir.replace("_single_char", "_multi_char")
            else:
                multi_char_dir = f"{single_char_dir}_multi_char"
            multi_char_path = prompt_path_obj.parent.parent / multi_char_dir / "prompt.md"
            if multi_char_path.exists():
                with open(multi_char_path, "r", encoding="utf-8") as f:
                    self.multi_char_prompt = f.read()
            else:
                # Fallback to single char prompt if multi-char doesn't exist
                self.multi_char_prompt = self.single_char_prompt
        else:
            # Default prompts if none provided
            self.single_char_prompt = self._get_default_prompt()
            self.multi_char_prompt = self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default formatting prompt."""
        return """You are a Chinese language learning assistant. Format the given field content to be clear, consistent, and well-structured for Anki flashcard study.

Guidelines:
- Keep the original meaning intact
- Use clear, consistent formatting
- Remove unnecessary repetition
- Use proper HTML markup where appropriate
- Make it suitable for language learning

Input format: You will receive the Chinese word/character and the original field content.
Output: Return only the formatted field content, no additional text."""

    def _calculate_cost(self, usage_dict: dict) -> float:
        """Calculate cost in USD based on token usage."""
        if self.model not in self.PRICING:
            return 0.0

        pricing = self.PRICING[self.model]
        prompt_tokens = usage_dict.get("prompt_tokens", 0)
        completion_tokens = usage_dict.get("completion_tokens", 0)

        # Cost per 1M tokens
        input_cost = (prompt_tokens * pricing["input"]) / 1_000_000
        output_cost = (completion_tokens * pricing["output"]) / 1_000_000

        return input_cost + output_cost

    def _is_single_character(self, chinese: str) -> bool:
        """Check if the Chinese text is a single character."""
        import re
        # Remove any non-Chinese characters and check if exactly one character remains
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', chinese)
        return len(chinese_chars) == 1

    def format_field(self, hanzi: str, original_content: str) -> FieldFormattingResult:
        """Format field content using GPT."""
        # Choose appropriate prompt based on character count
        is_single_char = self._is_single_character(hanzi)
        prompt_to_use = self.single_char_prompt if is_single_char else self.multi_char_prompt
        
        # Create the user message
        user_message = f"Chinese word: {hanzi}\nOriginal content: {original_content}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_to_use},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the formatted content
            formatted_content = response.choices[0].message.content.strip()
            
            # Calculate usage and cost
            usage_dict = response.usage.model_dump()
            cost = self._calculate_cost(usage_dict)
            
            token_usage = TokenUsage(
                prompt_tokens=usage_dict.get("prompt_tokens", 0),
                completion_tokens=usage_dict.get("completion_tokens", 0),
                total_tokens=usage_dict.get("total_tokens", 0),
                cost_usd=cost,
            )

            return FieldFormattingResult(
                formatted_content=formatted_content,
                token_usage=token_usage,
            )
            
        except Exception as e:
            # Return original content if formatting fails
            return FieldFormattingResult(
                formatted_content=original_content,
                token_usage=TokenUsage(),
            )


class GeminiFieldFormatter(FieldFormatter):
    """Format fields using Google Gemini models."""
    
    # Gemini pricing per 1M tokens
    PRICING = {
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.3125},
        "gemini-2.5-flash": {"input": 0.05, "output": 0.20, "cached_input": 0.0125},
        "gemini-2.5-flash-lite": {"input": 0.025, "output": 0.10, "cached_input": 0.00625},
    }

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        prompt_path: Optional[str] = None,
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = 500,
    ) -> None:
        import google.generativeai as genai
        from pathlib import Path

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load formatting prompts for both single and multi-character
        self.single_char_prompt = ""
        self.multi_char_prompt = ""
        
        if prompt_path:
            prompt_path_obj = Path(prompt_path)
            
            # Load single character prompt (the default one)
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.single_char_prompt = f.read()

            # Load multi-character prompt
            single_char_dir = prompt_path_obj.parent.name
            if single_char_dir.endswith("_single_char"):
                multi_char_dir = single_char_dir.replace("_single_char", "_multi_char")
            else:
                multi_char_dir = f"{single_char_dir}_multi_char"
            multi_char_path = prompt_path_obj.parent.parent / multi_char_dir / "prompt.md"
            if multi_char_path.exists():
                with open(multi_char_path, "r", encoding="utf-8") as f:
                    self.multi_char_prompt = f.read()
            else:
                # Fallback to single char prompt if multi-char doesn't exist
                self.multi_char_prompt = self.single_char_prompt
        else:
            # Default prompts if none provided
            self.single_char_prompt = self._get_default_prompt()
            self.multi_char_prompt = self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default formatting prompt."""
        return """You are a Chinese language learning assistant. Format the given field content to be clear, consistent, and well-structured for Anki flashcard study.

Guidelines:
- Keep the original meaning intact
- Use clear, consistent formatting
- Remove unnecessary repetition
- Use proper HTML markup where appropriate
- Make it suitable for language learning

Input format: You will receive the Chinese word/character and the original field content.
Output: Return only the formatted field content, no additional text."""

    def _calculate_cost(self, usage_dict: dict) -> float:
        """Calculate cost in USD based on token usage."""
        if self.model not in self.PRICING:
            return 0.0

        pricing = self.PRICING[self.model]
        prompt_tokens = usage_dict.get("prompt_token_count", 0)
        completion_tokens = usage_dict.get("candidates_token_count", 0)

        # Cost per 1M tokens
        input_cost = (prompt_tokens * pricing["input"]) / 1_000_000
        output_cost = (completion_tokens * pricing["output"]) / 1_000_000

        return input_cost + output_cost

    def _is_single_character(self, chinese: str) -> bool:
        """Check if the Chinese text is a single character."""
        import re
        # Remove any non-Chinese characters and check if exactly one character remains
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', chinese)
        return len(chinese_chars) == 1

    def format_field(self, hanzi: str, original_content: str) -> FieldFormattingResult:
        """Format field content using Gemini."""
        # Choose appropriate prompt based on character count
        is_single_char = self._is_single_character(hanzi)
        prompt_to_use = self.single_char_prompt if is_single_char else self.multi_char_prompt
        
        # Create the user message
        user_message = f"{prompt_to_use}\n\nChinese word: {hanzi}\nOriginal content: {original_content}"
        
        try:
            response = self.client.generate_content(
                user_message,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            
            # Extract the formatted content
            formatted_content = response.text.strip()
            
            # Calculate usage and cost
            usage_dict = {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
                "total_token_count": response.usage_metadata.total_token_count,
            }
            cost = self._calculate_cost(usage_dict)
            
            token_usage = TokenUsage(
                prompt_tokens=usage_dict.get("prompt_token_count", 0),
                completion_tokens=usage_dict.get("candidates_token_count", 0),
                total_tokens=usage_dict.get("total_token_count", 0),
                cost_usd=cost,
            )

            return FieldFormattingResult(
                formatted_content=formatted_content,
                token_usage=token_usage,
            )
            
        except Exception as e:
            # Return original content if formatting fails
            return FieldFormattingResult(
                formatted_content=original_content,
                token_usage=TokenUsage(),
            )