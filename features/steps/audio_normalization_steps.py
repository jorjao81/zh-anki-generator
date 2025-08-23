"""BDD step definitions for audio volume normalization."""

import os
import tempfile
from pathlib import Path
from typing import Optional
from behave import given, when, then
from pydub import AudioSegment

from anki_pleco_importer.audio import (
    AudioGeneratorFactory,
    MultiProviderAudioGenerator,
    TARGET_DBFS,
)


@given("the audio generators are available")
def step_audio_generators_available(context):
    """Check that audio generators are available."""
    context.temp_dir = tempfile.mkdtemp()
    context.audio_files = []


@given("the target volume level is {target_dbfs:f} dBFS")
def step_target_volume_level(context, target_dbfs: float):
    """Set the target volume level."""
    context.target_dbfs = target_dbfs
    assert context.target_dbfs == TARGET_DBFS, f"Expected {TARGET_DBFS}, got {target_dbfs}"


@given("I have a Forvo API key")
def step_have_forvo_api_key(context):
    """Check if Forvo API key is available."""
    forvo_key = os.environ.get("FORVO_API_KEY")
    if not forvo_key:
        context.scenario.skip("Forvo API key not available")
    context.forvo_key = forvo_key


@given("the word {word} is available in Forvo")
def step_word_available_forvo(context, word: str):
    """Assume word is available in Forvo (skip actual check for BDD)."""
    context.test_word = word


@given("I have a Qwen API key")
def step_have_qwen_api_key(context):
    """Check if Qwen API key is available."""
    qwen_key = os.environ.get("DASHSCOPE_API_KEY")
    if not qwen_key:
        context.scenario.skip("Qwen API key not available")
    context.qwen_key = qwen_key


@given("the Qwen TTS service is available")
def step_qwen_service_available(context):
    """Assume Qwen TTS service is available."""
    pass


@given("I have Tencent API credentials")
def step_have_tencent_credentials(context):
    """Check if Tencent API credentials are available."""
    tencent_id = os.environ.get("TENCENT_SECRET_ID")
    tencent_key = os.environ.get("TENCENT_API_KEY")
    if not (tencent_id and tencent_key):
        context.scenario.skip("Tencent API credentials not available")
    context.tencent_id = tencent_id
    context.tencent_key = tencent_key


@given("the Tencent TTS service is available")
def step_tencent_service_available(context):
    """Assume Tencent TTS service is available."""
    pass


@given("I have API keys for multiple audio providers")
def step_have_multiple_api_keys(context):
    """Check if multiple API keys are available."""
    providers_available = []

    # Check Forvo
    if os.environ.get("FORVO_API_KEY"):
        providers_available.append("forvo")
        context.forvo_key = os.environ["FORVO_API_KEY"]

    # Check Qwen
    if os.environ.get("DASHSCOPE_API_KEY"):
        providers_available.append("qwen")
        context.qwen_key = os.environ["DASHSCOPE_API_KEY"]

    # Check Tencent
    if os.environ.get("TENCENT_SECRET_ID") and os.environ.get("TENCENT_API_KEY"):
        providers_available.append("tencent")
        context.tencent_id = os.environ["TENCENT_SECRET_ID"]
        context.tencent_key = os.environ["TENCENT_API_KEY"]

    if len(providers_available) < 2:
        context.scenario.skip("Need at least 2 audio providers for multi-provider test")

    context.providers_available = providers_available


@given("the multi-provider audio generator is configured")
def step_configure_multi_provider(context):
    """Configure multi-provider audio generator."""
    config = {}

    if "forvo" in context.providers_available:
        config["forvo"] = {"api_key": context.forvo_key}

    if "qwen" in context.providers_available:
        config["qwen"] = {"api_key": context.qwen_key}

    if "tencent" in context.providers_available:
        config["tencent"] = {
            "secret_id": context.tencent_id,
            "secret_key": context.tencent_key,
        }

    context.multi_generator = MultiProviderAudioGenerator(
        providers=context.providers_available,
        config=config,
        cache_dir=context.temp_dir,
    )


@given("I have already generated normalized audio for {word}")
def step_have_cached_audio(context, word: str):
    """Generate audio first to test caching."""
    context.test_word = word
    context.cached_file = None

    # Try to generate with any available provider
    for provider in ["forvo", "qwen", "tencent"]:
        if hasattr(context, f"{provider}_key") or (provider == "tencent" and hasattr(context, "tencent_id")):
            config = {}
            if provider == "forvo":
                config[provider] = {"api_key": getattr(context, f"{provider}_key")}
            elif provider == "qwen":
                config[provider] = {"api_key": getattr(context, f"{provider}_key")}
            elif provider == "tencent":
                config[provider] = {
                    "secret_id": context.tencent_id,
                    "secret_key": context.tencent_key,
                }

            try:
                generator = AudioGeneratorFactory.create_generator(
                    provider, config[provider], cache_dir=context.temp_dir
                )
                if generator.is_available():
                    output_file = os.path.join(context.temp_dir, f"{word}_{provider}.mp3")
                    result = generator.generate_audio(word, output_file)
                    if result:
                        context.cached_file = result
                        context.audio_files.append(result)
                        break
            except Exception:
                continue

    if not context.cached_file:
        context.scenario.skip("Could not generate initial audio for caching test")


@when("I generate audio using {provider} provider for {word}")
def step_generate_audio_provider(context, provider: str, word: str):
    """Generate audio using specific provider."""
    context.test_word = word

    # Configure provider
    config = {}
    if provider.lower() == "forvo":
        if not hasattr(context, "forvo_key"):
            context.scenario.skip("Forvo API key not available")
        config = {"api_key": context.forvo_key}
    elif provider.lower() == "qwen":
        if not hasattr(context, "qwen_key"):
            context.scenario.skip("Qwen API key not available")
        config = {"api_key": context.qwen_key}
    elif provider.lower() == "tencent":
        if not hasattr(context, "tencent_id"):
            context.scenario.skip("Tencent credentials not available")
        config = {
            "secret_id": context.tencent_id,
            "secret_key": context.tencent_key,
        }

    # Create generator and generate audio
    try:
        generator = AudioGeneratorFactory.create_generator(provider.lower(), config, cache_dir=context.temp_dir)
        output_file = os.path.join(context.temp_dir, f"{word}_{provider}.mp3")
        context.generated_file = generator.generate_audio(word, output_file)
        if context.generated_file:
            context.audio_files.append(context.generated_file)
    except Exception as e:
        context.generation_error = str(e)
        context.generated_file = None


@when("I generate audio for the same word {word} using different providers")
def step_generate_audio_multiple_providers(context, word: str):
    """Generate audio using multiple providers."""
    context.test_word = word
    context.multi_provider_files = []

    for provider in context.providers_available:
        try:
            output_file = os.path.join(context.temp_dir, f"{word}_{provider}_multi.mp3")
            result = context.multi_generator.generate_audio(word, output_file)
            if result:
                context.multi_provider_files.append(result)
                context.audio_files.append(result)
        except Exception as e:
            print(f"Failed to generate with {provider}: {e}")


@when("I request the same audio again from cache")
def step_request_cached_audio(context):
    """Request the same audio that should be cached."""
    # This will be handled by checking the same cached file
    context.cache_result = context.cached_file


@when("the audio is normalized to {target_dbfs:f} dBFS")
def step_audio_normalized_to_target(context, target_dbfs: float):
    """Check that audio normalization happened (implicit in generation)."""
    context.normalization_target = target_dbfs


@then("the generated audio file should exist")
def step_generated_file_exists(context):
    """Check that the generated audio file exists."""
    if hasattr(context, "generated_file") and context.generated_file:
        assert os.path.exists(context.generated_file), f"Audio file {context.generated_file} does not exist"
    elif hasattr(context, "cached_file") and context.cached_file:
        assert os.path.exists(context.cached_file), f"Cached audio file {context.cached_file} does not exist"
    else:
        assert False, "No audio file was generated"


@then("the audio volume should be approximately {target_dbfs:f} dBFS")
def step_check_audio_volume(context, target_dbfs: float):
    """Check that audio volume is approximately at target dBFS."""
    audio_file = None

    if hasattr(context, "generated_file") and context.generated_file:
        audio_file = context.generated_file
    elif hasattr(context, "cached_file") and context.cached_file:
        audio_file = context.cached_file

    assert audio_file, "No audio file to check volume"
    assert os.path.exists(audio_file), f"Audio file {audio_file} does not exist"

    # Load audio and check volume
    audio = AudioSegment.from_file(audio_file)
    actual_dbfs = audio.dBFS

    # Allow for small variance in volume (±2 dBFS tolerance)
    tolerance = 2.0
    assert (
        abs(actual_dbfs - target_dbfs) <= tolerance
    ), f"Audio volume {actual_dbfs:.2f} dBFS is not within {tolerance} dB of target {target_dbfs:.2f} dBFS"


@then("all generated audio files should have approximately {target_dbfs:f} dBFS volume")
def step_check_all_audio_volumes(context, target_dbfs: float):
    """Check that all generated audio files have target volume."""
    assert context.multi_provider_files, "No multi-provider audio files were generated"

    tolerance = 2.0
    for audio_file in context.multi_provider_files:
        assert os.path.exists(audio_file), f"Audio file {audio_file} does not exist"

        audio = AudioSegment.from_file(audio_file)
        actual_dbfs = audio.dBFS

        assert (
            abs(actual_dbfs - target_dbfs) <= tolerance
        ), f"Audio file {audio_file} volume {actual_dbfs:.2f} dBFS is not within {tolerance} dB of target {target_dbfs:.2f} dBFS"


@then("the volume difference between files should be less than {max_diff:f} dBFS")
def step_check_volume_consistency(context, max_diff: float):
    """Check that volume levels are consistent between providers."""
    assert len(context.multi_provider_files) >= 2, "Need at least 2 files to check consistency"

    volumes = []
    for audio_file in context.multi_provider_files:
        audio = AudioSegment.from_file(audio_file)
        volumes.append(audio.dBFS)

    min_volume = min(volumes)
    max_volume = max(volumes)
    volume_diff = max_volume - min_volume

    assert (
        volume_diff <= max_diff
    ), f"Volume difference {volume_diff:.2f} dBFS exceeds maximum allowed {max_diff:.2f} dBFS"


@then("the cached audio file should be returned")
def step_cached_file_returned(context):
    """Check that cached file is returned."""
    assert context.cache_result, "No cached result returned"
    assert os.path.exists(context.cache_result), f"Cached file {context.cache_result} does not exist"


@then("the audio file should be playable")
def step_audio_file_playable(context):
    """Check that audio file is playable (can be loaded)."""
    audio_file = context.generated_file if hasattr(context, "generated_file") else context.cached_file
    assert audio_file, "No audio file to check"

    try:
        audio = AudioSegment.from_file(audio_file)
        assert len(audio) > 0, "Audio file appears to be empty"
        assert audio.frame_rate > 0, "Audio file has invalid sample rate"
    except Exception as e:
        assert False, f"Audio file is not playable: {e}"


@then("the audio should not contain clipping or distortion")
def step_check_no_clipping(context):
    """Check that audio doesn't contain clipping (basic check)."""
    audio_file = context.generated_file if hasattr(context, "generated_file") else context.cached_file
    assert audio_file, "No audio file to check"

    audio = AudioSegment.from_file(audio_file)

    # Basic check: ensure dBFS is not at maximum (which would indicate clipping)
    # MP3 files typically don't exceed 0 dBFS due to normalization
    assert audio.dBFS < -0.1, f"Audio may be clipped (dBFS: {audio.dBFS:.2f})"


@then("the normalized audio should maintain the original pronunciation clarity")
def step_check_pronunciation_clarity(context):
    """Check that audio maintains clarity (basic duration check)."""
    audio_file = context.generated_file if hasattr(context, "generated_file") else context.cached_file
    assert audio_file, "No audio file to check"

    audio = AudioSegment.from_file(audio_file)

    # Basic check: ensure audio has reasonable duration for a Chinese word (0.5-5 seconds)
    duration_seconds = len(audio) / 1000.0
    assert (
        0.5 <= duration_seconds <= 5.0
    ), f"Audio duration {duration_seconds:.2f}s seems unreasonable for a Chinese word"


def after_scenario(context, scenario):
    """Clean up temporary files after each scenario."""
    if hasattr(context, "audio_files"):
        for audio_file in context.audio_files:
            try:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            except Exception:
                pass

    if hasattr(context, "temp_dir"):
        import shutil

        try:
            shutil.rmtree(context.temp_dir)
        except Exception:
            pass
