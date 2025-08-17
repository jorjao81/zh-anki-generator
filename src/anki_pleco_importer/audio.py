"""Forvo audio generation for Chinese pronunciation."""

import os
import hashlib
import tempfile
import platform
import subprocess
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

import requests
import dashscope
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)


class AudioGeneratorError(Exception):
    """Base exception for audio generation errors."""

    pass


class TTSProviderNotAvailable(AudioGeneratorError):
    """Raised when a TTS provider is not available or configured."""

    pass


class AudioGenerator(ABC):
    """Abstract base class for audio generators."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("audio_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_filename(self, text: str, provider: str, details: Optional[str] = None) -> Path:
        """Generate a cache filename based on text, provider, and optional details."""
        # Sanitize the Chinese text for filename (remove unsafe characters)
        safe_text = re.sub(r'[<>:"/\\|?*]', "", text)
        safe_text = safe_text.replace(" ", "_")

        # Build filename components
        filename_parts = [safe_text, provider]

        # Add details if provided (e.g., Forvo username)
        if details:
            safe_details = re.sub(r'[<>:"/\\|?*]', "", details)
            safe_details = safe_details.replace(" ", "_")
            filename_parts.append(safe_details)

        # Add hash to avoid conflicts and ensure uniqueness
        content_hash = hashlib.md5(f"{text}_{provider}_{details or ''}".encode("utf-8")).hexdigest()[:8]
        filename_parts.append(content_hash)

        filename = "_".join(filename_parts) + ".mp3"
        return self.cache_dir / filename

    def _is_cached(self, text: str, provider: str, details: Optional[str] = None) -> bool:
        """Check if audio is already cached."""
        return self._get_cache_filename(text, provider, details).exists()

    def _get_cached_path(self, text: str, provider: str, details: Optional[str] = None) -> Optional[str]:
        """Get path to cached audio file."""
        cache_file = self._get_cache_filename(text, provider, details)
        return str(cache_file) if cache_file.exists() else None

    @abstractmethod
    def generate_audio(self, text: str, output_file: str) -> Optional[str]:
        """Generate audio for the given text."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this audio generator is available."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        pass

    def generate_with_cache(
        self,
        text: str,
        output_file: Optional[str] = None,
        cache_details: Optional[str] = None,
    ) -> Optional[str]:
        """Generate audio with caching support."""
        provider = self.get_provider_name()

        # Check cache first
        if self._is_cached(text, provider, cache_details):
            cached_path = self._get_cached_path(text, provider, cache_details)
            logger.info(f"Using cached audio for '{text}' from {provider}")

            # If output_file is specified, copy from cache
            if output_file and cached_path:
                import shutil

                shutil.copy2(cached_path, output_file)
                return output_file
            return cached_path

        # Generate new audio
        if not output_file:
            output_file = str(self._get_cache_filename(text, provider, cache_details))

        result = self.generate_audio(text, output_file)

        # Cache the result if successful and not already in cache location
        if result and result != str(self._get_cache_filename(text, provider, cache_details)):
            cache_file = self._get_cache_filename(text, provider, cache_details)
            import shutil

            shutil.copy2(result, cache_file)

        return result


class ForvoGenerator(AudioGenerator):
    """Forvo community pronunciation generator with smart user selection."""

    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[str] = None,
        use_paid_api: bool = True,
        preferred_users: Optional[List[str]] = None,
        download_all_when_no_preferred: bool = True,
        interactive_selection: bool = True,
    ):
        super().__init__(cache_dir)
        self.api_key = api_key
        self.language = "zh"
        # Note: Forvo uses the same endpoint for both free and paid APIs
        # The difference is in rate limits and features, not the URL
        self.base_url = "https://apifree.forvo.com"
        self.use_paid_api = use_paid_api
        self.preferred_users = preferred_users or []
        self.download_all_when_no_preferred = download_all_when_no_preferred
        self.interactive_selection = interactive_selection
        self._cached_selection: Optional[Dict[str, Any]] = None

    def is_available(self) -> bool:
        """Check if Forvo API is available."""
        return bool(self.api_key)

    def get_provider_name(self) -> str:
        return "forvo"

    def _find_cached_forvo_audio(self, text: str) -> Optional[str]:
        """Find any cached Forvo audio for this text, regardless of username."""
        # Look for any existing Forvo cache files for this text
        safe_text = re.sub(r'[<>:"/\\|?*]', "", text).replace(" ", "_")
        pattern = f"{safe_text}_forvo_*.mp3"
        cache_files = list(self.cache_dir.glob(pattern))
        return str(cache_files[0]) if cache_files else None

    def _format_pronunciation_info(self, pronunciation: Dict) -> str:
        """Format pronunciation information for display."""
        username = pronunciation.get("username", "unknown")
        sex = pronunciation.get("sex", "?")
        country = pronunciation.get("country", "unknown")
        votes = pronunciation.get("num_votes", 0)
        # positive_votes = pronunciation.get("num_positive_votes", 0)
        rating = pronunciation.get("rate", 0)

        # Colored gender icons
        if sex == "f":
            gender_icon = "\033[95m♀\033[0m"  # Magenta/pink for female
        elif sex == "m":
            gender_icon = "\033[94m♂\033[0m"  # Blue for male
        else:
            gender_icon = "\033[90m?\033[0m"  # Gray for unknown

        # Country flags mapping
        country_flags = {
            "China": "🇨🇳",
            "Taiwan": "🇹🇼",
            "Hong Kong": "🇭🇰",
            "Singapore": "🇸🇬",
            "United States": "🇺🇸",
            "Canada": "🇨🇦",
            "United Kingdom": "🇬🇧",
            "Australia": "🇦🇺",
            "New Zealand": "🇳🇿",
            "Japan": "🇯🇵",
            "South Korea": "🇰🇷",
            "Thailand": "🇹🇭",
            "Malaysia": "🇲🇾",
            "Philippines": "🇵🇭",
            "Indonesia": "🇮🇩",
            "Vietnam": "🇻🇳",
            "Germany": "🇩🇪",
            "France": "🇫🇷",
            "Spain": "🇪🇸",
            "Italy": "🇮🇹",
            "Netherlands": "🇳🇱",
            "Brazil": "🇧🇷",
            "Mexico": "🇲🇽",
            "Argentina": "🇦🇷",
            "Russia": "🇷🇺",
            "India": "🇮🇳",
        }

        # Get country flag or fallback to country name
        country_display = country_flags.get(country, f"🏳️ {country}")

        # Rating stars
        stars = "⭐" * min(5, max(0, int(rating))) if rating > 0 else ""

        # Preferred user indicator
        preferred = " [PREFERRED]" if username in self.preferred_users else ""

        return f"{username} ({gender_icon} {country_display}) - {votes} votes, rating: {rating:.1f} {stars}{preferred}"

    def _select_best_pronunciation(self, pronunciations: List[Dict], text: str) -> Optional[Dict]:
        """Select the best pronunciation based on preferences."""
        if not pronunciations:
            return None

        # Check for preferred users in order - auto-select only if found
        for preferred_user in self.preferred_users:
            for pronunciation in pronunciations:
                if pronunciation.get("username") == preferred_user:
                    logger.info(f"Auto-selecting preferred user '{preferred_user}' for '{text}'")
                    return pronunciation

        # No preferred users found - always show interactive selection if enabled
        if self.interactive_selection:
            return self._interactive_pronunciation_selection(pronunciations, text)

        # Interactive selection disabled - use highest-rated pronunciation
        if not self.download_all_when_no_preferred:
            best = max(
                pronunciations,
                key=lambda x: (x.get("num_positive_votes", 0), x.get("num_votes", 0)),
            )
            logger.info(f"No preferred users found for '{text}', using highest-rated: {best.get('username')}")
            return best
        else:
            # Just use the first one
            return pronunciations[0]

    def _play_audio(self, file_path: str) -> bool:
        """Play audio file with cross-platform support."""
        try:
            # Try playsound3 first
            try:
                from playsound3 import playsound

                playsound(file_path)
                return True
            except ImportError:
                logger.debug("playsound3 not available, falling back to system commands")

            # Fallback to system commands
            system = platform.system()

            if system == "Darwin":  # macOS
                subprocess.run(["afplay", file_path], check=True, capture_output=True)
                return True
            elif system == "Linux":
                # Try mpg123 first, then mpv
                for player, args in [
                    ("mpg123", ["-q"]),
                    ("mpv", ["--no-video"]),
                    ("ffplay", ["-nodisp", "-autoexit"]),
                ]:
                    try:
                        subprocess.run(
                            [player] + args + [file_path],
                            check=True,
                            capture_output=True,
                        )
                        return True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            elif system == "Windows":
                # Try ffplay if available
                try:
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", file_path],
                        check=True,
                        capture_output=True,
                    )
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

            return False

        except Exception as e:
            logger.warning(f"Failed to play audio: {e}")
            return False

    def _download_pronunciation_preview(self, pronunciation: Dict, text: str) -> Optional[str]:
        """Download pronunciation to temporary file for preview."""
        audio_url = pronunciation.get("pathmp3")
        if not audio_url:
            return None

        try:
            # Create temporary file
            username = pronunciation.get("username", "unknown")
            temp_file = tempfile.NamedTemporaryFile(suffix=f"_{username}_{text[:10]}.mp3", delete=False)
            temp_file.close()

            # Download audio
            logger.debug(f"Downloading preview audio from: {audio_url}")
            audio_response = requests.get(audio_url, timeout=30)
            audio_response.raise_for_status()

            with open(temp_file.name, "wb") as f:
                f.write(audio_response.content)

            return temp_file.name

        except Exception as e:
            logger.warning(f"Failed to download preview audio: {e}")
            return None

    def _cleanup_preview_files(self, preview_files: List[str]) -> None:
        """Clean up temporary preview files."""
        for file_path in preview_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup preview file {file_path}: {e}")

    def _interactive_pronunciation_selection(self, pronunciations: List[Dict], text: str) -> Optional[Dict]:
        """Allow interactive selection of pronunciation with audio preview."""
        print(f"\n🎵 {text} - Found {len(pronunciations)} pronunciations:")

        for i, pronunciation in enumerate(pronunciations, 1):
            info = self._format_pronunciation_info(pronunciation)
            print(f"{i:2d}. {info}")

        print("\nCommands: <number> to play, s<number> to select, 's' to skip")
        print("Example: '1' to play option 1, 's1' to select option 1, 's' to skip all")

        preview_files = []  # Track temporary files for cleanup

        try:
            while True:
                try:
                    choice = input("\nChoice: ").strip().lower()

                    if choice == "s":
                        logger.info(f"User skipped pronunciation selection for '{text}'")
                        return None

                    # Handle selection commands (s1, s2, etc.)
                    if choice.startswith("s") and len(choice) > 1:
                        try:
                            select_num = int(choice[1:])
                            if 1 <= select_num <= len(pronunciations):
                                selected = pronunciations[select_num - 1]
                                username = selected.get("username", "unknown")
                                logger.info(f"User selected pronunciation by '{username}' for '{text}'")
                                return selected
                            else:
                                print(f"Please enter a number between 1 and {len(pronunciations)}")
                        except ValueError:
                            print("Invalid selection command. Use format: s1, s2, etc.")
                        continue

                    # Handle play commands (1, 2, etc.)
                    try:
                        play_num = int(choice)
                        if 1 <= play_num <= len(pronunciations):
                            pronunciation = pronunciations[play_num - 1]
                            username = pronunciation.get("username", "unknown")

                            print(f"🔊 Downloading and playing pronunciation by {username}...")

                            # Download to temporary file
                            temp_file = self._download_pronunciation_preview(pronunciation, text)
                            if temp_file:
                                preview_files.append(temp_file)

                                # Try to play the audio
                                if self._play_audio(temp_file):
                                    print(f"✅ Played pronunciation by {username}")
                                else:
                                    print("❌ Could not play audio (file downloaded but playback failed)")
                                    print("You may need to install an audio player or playsound3")
                            else:
                                print("❌ Could not download audio for preview")
                        else:
                            print(f"Please enter a number between 1 and {len(pronunciations)}")
                    except ValueError:
                        print("Invalid input. Use: number to play, s<number> to select, 's' to skip")

                except KeyboardInterrupt:
                    print("\nSkipping pronunciation selection...")
                    return None
        finally:
            # Clean up temporary preview files
            if preview_files:
                self._cleanup_preview_files(preview_files)

    def generate_audio(self, text: str, output_file: str) -> Optional[str]:
        """Download audio from Forvo with smart user selection."""
        if not self.is_available():
            raise TTSProviderNotAvailable("Forvo API not available")

        try:
            # Get pronunciation URL
            url = (
                f"{self.base_url}/key/{self.api_key}/format/json/"
                f"action/word-pronunciations/word/{text}/language/{self.language}"
            )
            logger.info(f"Forvo API request: {url}")

            response = requests.get(url, timeout=10)
            logger.info(f"Forvo API response status: {response.status_code}")

            if response.status_code == 401:
                logger.error("Forvo API authentication failed - check your API key")
                return None
            elif response.status_code == 403:
                logger.error("Forvo API access forbidden - check your subscription status")
                return None
            elif response.status_code == 429:
                logger.error("Forvo API rate limit exceeded")
                return None

            response.raise_for_status()

            data = response.json()

            # Check for API error messages
            if "error" in data:
                logger.error(f"Forvo API error: {data['error']}")
                return None

            pronunciations = data.get("items", [])
            if not pronunciations:
                logger.warning(f"No Forvo pronunciation items found for '{text}'")
                return None

            # Use cached selection if available, otherwise select
            selected_pronunciation: Optional[Dict[str, Any]] = None
            if hasattr(self, "_cached_selection") and self._cached_selection:
                selected_pronunciation = self._cached_selection
                # Clear the cache after using it
                self._cached_selection = None
                logger.info(f"Using cached pronunciation selection for '{text}'")
            else:
                selected_pronunciation = self._select_best_pronunciation(pronunciations, text)

            if not selected_pronunciation:
                logger.info(f"No pronunciation selected for '{text}'")
                return None

            # Get user info for logging and filename
            username = selected_pronunciation.get("username", "unknown")
            audio_url = selected_pronunciation.get("pathmp3")

            if not audio_url:
                logger.warning(f"No audio URL found in selected pronunciation for '{text}'")
                return None

            logger.info(f"Downloading audio from: {audio_url}")
            logger.info(f"Selected pronunciation by: {username}")

            # Download the audio file
            audio_response = requests.get(audio_url, timeout=30)
            audio_response.raise_for_status()

            # Write to output file
            with open(output_file, "wb") as f:
                f.write(audio_response.content)

            logger.info(f"Forvo downloaded audio for '{text}' by '{username}' to {output_file}")
            return output_file

        except requests.exceptions.RequestException as e:
            logger.error(f"Forvo network error: {e}")
            return None
        except Exception as e:
            logger.error(f"Forvo unexpected error: {e}")
            return None

    def generate_with_cache(
        self,
        text: str,
        output_file: Optional[str] = None,
        cache_details: Optional[str] = None,
    ) -> Optional[str]:
        """Generate audio with Forvo-specific caching that includes username."""
        # Check for any existing cached audio first
        cached_audio = self._find_cached_forvo_audio(text)
        if cached_audio:
            logger.info(f"Using cached Forvo audio for '{text}': {cached_audio}")
            if output_file and output_file != cached_audio:
                import shutil

                shutil.copy2(cached_audio, output_file)
                return output_file
            return cached_audio

        # No cache found, need to download and determine username first
        try:
            # Get pronunciation URL to determine username
            url = (
                f"{self.base_url}/key/{self.api_key}/format/json/"
                f"action/word-pronunciations/word/{text}/language/{self.language}"
            )
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()
            pronunciations = data.get("items", [])
            if not pronunciations:
                return None

            # Select the best pronunciation to get username - cache the result
            selected_pronunciation = self._select_best_pronunciation(pronunciations, text)
            if not selected_pronunciation:
                return None

            # Store the selection to avoid re-prompting
            self._cached_selection = selected_pronunciation
            username = selected_pronunciation.get("username", "unknown")

            # Use new caching system with username as cache_details
            return super().generate_with_cache(text, output_file, username)

        except Exception as e:
            logger.error(f"Forvo cache lookup error: {e}")
            return None


class QwenGenerator(AudioGenerator):
    """Qwen TTS audio generator with multiple voice options."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        voices: Optional[List[str]] = None,
    ):
        super().__init__(cache_dir)
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.voices = voices or ["Chelsie", "Cherry", "Ethan", "Serena"]
        
        # Set the API key for dashscope
        if self.api_key:
            dashscope.api_key = self.api_key

    def is_available(self) -> bool:
        """Check if Qwen TTS is available."""
        return bool(self.api_key)

    def get_provider_name(self) -> str:
        return "qwen"

    def generate_audio(self, text: str, output_file: str) -> Optional[str]:
        """Generate audio using Qwen TTS with a single voice."""
        if not self.is_available():
            raise TTSProviderNotAvailable("Qwen TTS API not available")

        try:
            # Extract voice from output filename if it contains voice info
            voice = "Cherry"  # default
            for v in self.voices:
                if v.lower() in output_file.lower():
                    voice = v
                    break

            logger.info(f"Generating Qwen audio for '{text}' with voice '{voice}'")

            response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                model="qwen-tts",
                text=text,
                voice=voice,
            )

            if not response.output or not response.output.audio:
                logger.error(f"Qwen TTS failed for '{text}': No output audio")
                return None

            audio_url = response.output.audio["url"]
            
            # Download audio content
            audio_response = requests.get(audio_url, timeout=30)
            audio_response.raise_for_status()

            # Convert WAV data to MP3 in memory
            audio_data = io.BytesIO(audio_response.content)
            audio = AudioSegment.from_wav(audio_data)
            audio.export(output_file, format="mp3")

            logger.info(f"Qwen audio generated for '{text}' with voice '{voice}' to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Qwen TTS error for '{text}': {e}")
            return None

    def generate_all_voices(self, text: str, base_output_dir: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Generate audio for all voices and return a mapping of voice -> file path."""
        results = {}
        
        for voice in self.voices:
            # Create filename using the specified format: WORDINCHINESE-qwen-VOICE.mp3
            if base_output_dir:
                output_file = os.path.join(base_output_dir, f"{text}-qwen-{voice}.mp3")
            else:
                output_file = str(self.cache_dir / f"{text}-qwen-{voice}.mp3")
            
            # Check cache first
            if self._is_cached(text, voice):
                cached_path = self._get_cached_path(text, voice)
                if cached_path:
                    results[voice] = cached_path
                    logger.info(f"Using cached Qwen audio for '{text}' voice '{voice}'")
                    continue
            
            try:
                logger.info(f"Generating Qwen audio for '{text}' with voice '{voice}'")

                response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                    model="qwen-tts",
                    text=text,
                    voice=voice,
                )

                if not response.output or not response.output.audio:
                    logger.error(f"Qwen TTS failed for '{text}' voice '{voice}': No output audio")
                    results[voice] = None
                    continue

                audio_url = response.output.audio["url"]
                
                # Download audio content
                audio_response = requests.get(audio_url, timeout=30)
                audio_response.raise_for_status()

                # Convert WAV data to MP3 in memory
                audio_data = io.BytesIO(audio_response.content)
                audio = AudioSegment.from_wav(audio_data)
                audio.export(output_file, format="mp3")

                results[voice] = output_file
                logger.info(f"Qwen audio generated for '{text}' voice '{voice}' to {output_file}")

            except Exception as e:
                logger.error(f"Qwen TTS error for '{text}' voice '{voice}': {e}")
                results[voice] = None

        return results


class ForvoWithQwenFallbackGenerator(AudioGenerator):
    """Forvo generator with Qwen TTS fallback when no preferred pronouncer is found."""

    def __init__(
        self,
        forvo_config: Dict[str, Any],
        qwen_config: Dict[str, Any],
        cache_dir: Optional[str] = None,
        enable_qwen_fallback: bool = True,
    ):
        super().__init__(cache_dir)
        self.enable_qwen_fallback = enable_qwen_fallback
        
        # Initialize Forvo generator
        self.forvo_generator = ForvoGenerator(
            api_key=forvo_config.get("api_key"),
            cache_dir=cache_dir,
            use_paid_api=forvo_config.get("use_paid_api", True),
            preferred_users=forvo_config.get("preferred_users", []),
            download_all_when_no_preferred=False,  # We'll handle this ourselves
            interactive_selection=forvo_config.get("interactive_selection", True),
        )
        
        # Initialize Qwen generator if fallback is enabled
        self.qwen_generator = None
        if self.enable_qwen_fallback:
            self.qwen_generator = QwenGenerator(
                api_key=qwen_config.get("api_key"),
                cache_dir=cache_dir,
                voices=qwen_config.get("voices", ["Chelsie", "Cherry", "Ethan", "Serena"]),
            )

    def is_available(self) -> bool:
        """Check if at least Forvo is available."""
        return self.forvo_generator.is_available()

    def get_provider_name(self) -> str:
        return "forvo_qwen"

    def _has_preferred_pronouncer(self, text: str) -> bool:
        """Check if Forvo has a preferred pronouncer for the given text."""
        if not self.forvo_generator.preferred_users:
            return False
            
        try:
            # Get pronunciation URL to check for preferred users
            url = (
                f"{self.forvo_generator.base_url}/key/{self.forvo_generator.api_key}/format/json/"
                f"action/word-pronunciations/word/{text}/language/{self.forvo_generator.language}"
            )
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return False

            data = response.json()
            pronunciations = data.get("items", [])
            
            # Check if any preferred users are available
            for pronunciation in pronunciations:
                if pronunciation.get("username") in self.forvo_generator.preferred_users:
                    return True
                    
            return False

        except Exception as e:
            logger.warning(f"Failed to check for preferred pronouncer: {e}")
            return False

    def _get_enhanced_pronunciation_options(self, text: str) -> List[Dict[str, Any]]:
        """Get all pronunciation options including Forvo and Qwen voices."""
        options = []
        
        # Get Forvo pronunciations
        try:
            url = (
                f"{self.forvo_generator.base_url}/key/{self.forvo_generator.api_key}/format/json/"
                f"action/word-pronunciations/word/{text}/language/{self.forvo_generator.language}"
            )
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                forvo_pronunciations = data.get("items", [])
                
                # Add Forvo pronunciations with type indicator
                for pronunciation in forvo_pronunciations:
                    pronunciation["source"] = "forvo"
                    options.append(pronunciation)
                    
        except Exception as e:
            logger.warning(f"Failed to get Forvo pronunciations: {e}")

        # Add Qwen voices if available and fallback is enabled
        if self.enable_qwen_fallback and self.qwen_generator and self.qwen_generator.is_available():
            for voice in self.qwen_generator.voices:
                qwen_option = {
                    "source": "qwen",
                    "voice": voice,
                    "username": f"Qwen-{voice}",
                    "pathmp3": None,  # Will be generated on demand
                    "text": text,
                }
                options.append(qwen_option)

        return options

    def _enhanced_interactive_selection(self, options: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
        """Enhanced interactive selection including Qwen voices."""
        if not options:
            return None
            
        print(f"\n🎵 {text} - Found {len(options)} pronunciation options:")

        # Display options with enhanced formatting
        for i, option in enumerate(options, 1):
            if option["source"] == "forvo":
                info = self.forvo_generator._format_pronunciation_info(option)
                print(f"{i:2d}. [Forvo] {info}")
            elif option["source"] == "qwen":
                voice = option["voice"]
                print(f"{i:2d}. [Qwen TTS] {voice} (AI-generated voice)")

        print("\nCommands: <number> to play, s<number> to select, 's' to skip")
        print("Example: '1' to play option 1, 's1' to select option 1, 's' to skip all")

        preview_files = []  # Track temporary files for cleanup

        try:
            while True:
                try:
                    choice = input("\nChoice: ").strip().lower()

                    if choice == "s":
                        logger.info(f"User skipped pronunciation selection for '{text}'")
                        return None

                    # Handle selection commands (s1, s2, etc.)
                    if choice.startswith("s") and len(choice) > 1:
                        try:
                            select_num = int(choice[1:])
                            if 1 <= select_num <= len(options):
                                selected = options[select_num - 1]
                                source = selected["source"]
                                if source == "forvo":
                                    username = selected.get("username", "unknown")
                                    logger.info(f"User selected Forvo pronunciation by '{username}' for '{text}'")
                                elif source == "qwen":
                                    voice = selected["voice"]
                                    logger.info(f"User selected Qwen voice '{voice}' for '{text}'")
                                return selected
                            else:
                                print(f"Please enter a number between 1 and {len(options)}")
                        except ValueError:
                            print("Invalid selection command. Use format: s1, s2, etc.")
                        continue

                    # Handle play commands (1, 2, etc.)
                    try:
                        play_num = int(choice)
                        if 1 <= play_num <= len(options):
                            option = options[play_num - 1]
                            
                            if option["source"] == "forvo":
                                username = option.get("username", "unknown")
                                print(f"🔊 Downloading and playing Forvo pronunciation by {username}...")
                                
                                # Download to temporary file using Forvo's method
                                temp_file = self.forvo_generator._download_pronunciation_preview(option, text)
                                if temp_file:
                                    preview_files.append(temp_file)
                                    if self.forvo_generator._play_audio(temp_file):
                                        print(f"✅ Played Forvo pronunciation by {username}")
                                    else:
                                        print("❌ Could not play audio (file downloaded but playback failed)")
                                else:
                                    print("❌ Could not download Forvo audio for preview")
                                    
                            elif option["source"] == "qwen":
                                voice = option["voice"]
                                print(f"🔊 Generating and playing Qwen voice {voice}...")
                                
                                # Generate Qwen audio using cache (so we don't call API twice)
                                cache_details = voice  # Just the voice name, provider is already added by _get_cache_filename
                                result = self.qwen_generator.generate_with_cache(text, cache_details=cache_details)
                                if result:
                                    if self.forvo_generator._play_audio(result):
                                        print(f"✅ Played Qwen voice {voice}")
                                    else:
                                        print("❌ Could not play audio (file generated but playback failed)")
                                else:
                                    print("❌ Could not generate Qwen audio for preview")
                        else:
                            print(f"Please enter a number between 1 and {len(options)}")
                    except ValueError:
                        print("Invalid input. Use: number to play, s<number> to select, 's' to skip")

                except KeyboardInterrupt:
                    print("\nSkipping pronunciation selection...")
                    return None
        finally:
            # Clean up temporary preview files
            if preview_files:
                self.forvo_generator._cleanup_preview_files(preview_files)

    def generate_audio(self, text: str, output_file: str) -> Optional[str]:
        """Generate audio with Forvo preferred user priority and Qwen fallback."""
        if not self.is_available():
            raise TTSProviderNotAvailable("Forvo API not available")

        # First, check if we have a preferred pronouncer in Forvo
        if self._has_preferred_pronouncer(text):
            logger.info(f"Found preferred Forvo pronouncer for '{text}', using Forvo")
            return self.forvo_generator.generate_audio(text, output_file)

        # No preferred pronouncer found, show enhanced selection with Qwen options
        if self.enable_qwen_fallback and self.forvo_generator.interactive_selection:
            options = self._get_enhanced_pronunciation_options(text)
            if options:
                selected = self._enhanced_interactive_selection(options, text)
                if selected:
                    if selected["source"] == "forvo":
                        # Use the selected Forvo pronunciation
                        self.forvo_generator._cached_selection = selected
                        return self.forvo_generator.generate_audio(text, output_file)
                    elif selected["source"] == "qwen":
                        # Generate with selected Qwen voice (use cache to avoid duplicate API calls)
                        voice = selected["voice"]
                        cache_details = voice  # Just the voice name, provider is already added by _get_cache_filename
                        # Ensure the output file includes the voice name for proper generation
                        voice_output_file = output_file.replace(".mp3", f"-qwen-{voice}.mp3")
                        return self.qwen_generator.generate_with_cache(text, voice_output_file, cache_details)
                else:
                    # User explicitly skipped - don't fall back, they already saw all options
                    logger.info(f"User skipped pronunciation selection for '{text}' - no fallback")
                    return None

        # If interactive selection is disabled, fall back to standard Forvo behavior
        logger.info(f"Interactive selection disabled for '{text}', using standard Forvo")
        return self.forvo_generator.generate_audio(text, output_file)


class AudioGeneratorFactory:
    """Factory for creating audio generators."""

    @staticmethod
    def create_generator(provider: str, config: Dict[str, Any], cache_dir: Optional[str] = None) -> AudioGenerator:
        """Create an audio generator based on provider and configuration."""
        if provider == "forvo":
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("Forvo API key is required")
            return ForvoGenerator(
                api_key=api_key,
                cache_dir=cache_dir,
                use_paid_api=config.get("use_paid_api", True),
                preferred_users=config.get("preferred_users", []),
                download_all_when_no_preferred=config.get("download_all_when_no_preferred", True),
                interactive_selection=config.get("interactive_selection", True),
            )
        elif provider == "qwen":
            api_key = config.get("api_key")
            return QwenGenerator(
                api_key=api_key,
                cache_dir=cache_dir,
                voices=config.get("voices", ["Chelsie", "Cherry", "Ethan", "Serena"]),
            )
        elif provider == "forvo_qwen":
            # This provider requires both forvo and qwen configs
            # Check if configs are nested under forvo_qwen or at top level
            if "forvo" in config and "qwen" in config:
                # Nested configuration
                forvo_config = config.get("forvo", {})
                qwen_config = config.get("qwen", {})
                enable_qwen_fallback = config.get("enable_qwen_fallback", True)
            else:
                # Top-level configuration - get from parent config
                # This will be passed from the CLI when it extracts forvo_qwen config
                forvo_config = config.get("forvo", {})
                qwen_config = config.get("qwen", {})
                enable_qwen_fallback = config.get("enable_qwen_fallback", True)
            
            return ForvoWithQwenFallbackGenerator(
                forvo_config=forvo_config,
                qwen_config=qwen_config,
                cache_dir=cache_dir,
                enable_qwen_fallback=enable_qwen_fallback,
            )
        else:
            raise ValueError(f"Unknown audio provider: {provider}")

    @staticmethod
    def get_available_providers(config: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get list of available providers based on configuration."""
        available = []

        for provider in ["forvo", "qwen", "forvo_qwen"]:
            try:
                generator = AudioGeneratorFactory.create_generator(provider, config.get(provider, {}), cache_dir=None)
                if generator.is_available():
                    available.append(provider)
            except Exception:
                pass

        return available


class MultiProviderAudioGenerator:
    """Audio generator that tries multiple providers in order of preference."""

    def __init__(
        self,
        providers: List[str],
        config: Dict[str, Dict[str, Any]],
        cache_dir: Optional[str] = None,
    ):
        self.providers = providers
        self.config = config
        self.cache_dir = cache_dir
        self.generators = {}
        self.skipped_words: List[str] = []  # Track words with no pronunciation selected

        # Initialize generators
        for provider in providers:
            try:
                generator = AudioGeneratorFactory.create_generator(provider, config.get(provider, {}), cache_dir)
                if generator.is_available():
                    self.generators[provider] = generator
                    logger.info(f"Audio provider '{provider}' is available")
                else:
                    logger.warning(f"Audio provider '{provider}' is not available")
            except Exception as e:
                logger.error(f"Failed to initialize audio provider '{provider}': {e}")

    def generate_audio(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """Generate audio using the first available provider."""
        for provider in self.providers:
            generator = self.generators.get(provider)
            if generator:
                try:
                    # Get cache details if the generator supports it
                    cache_details = None
                    if hasattr(generator, "get_cache_details"):
                        cache_details = generator.get_cache_details()

                    result = generator.generate_with_cache(text, output_file, cache_details)
                    if result:
                        logger.info(f"Successfully generated audio for '{text}' using {provider}")
                        return result
                    else:
                        # No result could mean user skipped or no pronunciation found
                        if text not in self.skipped_words:
                            self.skipped_words.append(text)
                        logger.info(f"No audio generated for '{text}' using {provider}")
                except Exception as e:
                    logger.error(f"Provider '{provider}' failed for '{text}': {e}")
                    continue

        # All providers failed - also track as skipped
        if text not in self.skipped_words:
            self.skipped_words.append(text)
        logger.error(f"All audio providers failed for '{text}'")
        return None

    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.generators.keys())

    def get_skipped_words(self) -> List[str]:
        """Get list of words for which no pronunciation was selected."""
        return self.skipped_words.copy()

    def clear_skipped_words(self) -> None:
        """Clear the list of skipped words."""
        self.skipped_words.clear()
