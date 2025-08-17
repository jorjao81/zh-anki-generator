#!/usr/bin/env python3
"""Test script for Qwen integration with Forvo audio generation."""

import os
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add src to path
import sys
sys.path.insert(0, 'src')

from anki_pleco_importer.audio import QwenGenerator, AudioGeneratorFactory

def test_qwen_generator():
    """Test Qwen generator basic functionality."""
    print("=" * 50)
    print("Testing Qwen Generator")
    print("=" * 50)
    
    # Check for API key
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("⚠️  DASHSCOPE_API_KEY environment variable not set")
        print("   Skipping Qwen tests")
        return False
    
    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test QwenGenerator
        config = {
            "api_key": api_key,
            "voices": ["Cherry", "Chelsie"]  # Test with subset for speed
        }
        
        generator = QwenGenerator(
            api_key=config["api_key"],
            cache_dir=temp_dir,
            voices=config["voices"]
        )
        
        if not generator.is_available():
            print("❌ Qwen generator not available")
            return False
        
        print("✅ Qwen generator is available")
        
        # Test single voice generation
        test_word = "你好"
        output_file = os.path.join(temp_dir, f"{test_word}-test.mp3")
        
        print(f"🔊 Generating audio for '{test_word}'...")
        result = generator.generate_audio(test_word, output_file)
        
        if result and os.path.exists(result):
            print(f"✅ Successfully generated audio: {result}")
            print(f"   File size: {os.path.getsize(result)} bytes")
        else:
            print("❌ Failed to generate audio")
            return False
        
        # Test all voices generation
        print(f"🔊 Generating all voices for '{test_word}'...")
        all_voices_result = generator.generate_all_voices(test_word, temp_dir)
        
        success_count = sum(1 for path in all_voices_result.values() if path and os.path.exists(path))
        print(f"✅ Generated {success_count}/{len(config['voices'])} voice files")
        
        for voice, path in all_voices_result.items():
            if path and os.path.exists(path):
                size = os.path.getsize(path)
                print(f"   {voice}: {Path(path).name} ({size} bytes)")
            else:
                print(f"   {voice}: ❌ Failed")
        
        return success_count > 0

def test_factory_integration():
    """Test AudioGeneratorFactory integration."""
    print("\n" + "=" * 50)
    print("Testing Factory Integration")
    print("=" * 50)
    
    # Test qwen provider
    config = {
        "qwen": {
            "api_key": os.environ.get('DASHSCOPE_API_KEY'),
            "voices": ["Cherry"]
        }
    }
    
    if not config["qwen"]["api_key"]:
        print("⚠️  DASHSCOPE_API_KEY not set, skipping factory test")
        return False
    
    try:
        # Test creating qwen generator through factory
        generator = AudioGeneratorFactory.create_generator("qwen", config["qwen"])
        print("✅ Factory can create Qwen generator")
        
        # Test getting available providers
        available = AudioGeneratorFactory.get_available_providers(config)
        print(f"✅ Available providers: {available}")
        
        return "qwen" in available
        
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False

def test_forvo_qwen_integration():
    """Test the integrated Forvo+Qwen provider."""
    print("\n" + "=" * 50)
    print("Testing Forvo+Qwen Integration")
    print("=" * 50)
    
    forvo_key = os.environ.get('FORVO_API_KEY')
    qwen_key = os.environ.get('DASHSCOPE_API_KEY')
    
    if not qwen_key:
        print("⚠️  DASHSCOPE_API_KEY not set, skipping integration test")
        return False
    
    if not forvo_key:
        print("⚠️  FORVO_API_KEY not set, will test Qwen-only fallback")
    
    config = {
        "forvo_qwen": {
            "forvo": {
                "api_key": forvo_key,
                "preferred_users": ["nonexistent_user"],  # Force fallback
                "interactive_selection": False  # Disable for automated testing
            },
            "qwen": {
                "api_key": qwen_key,
                "voices": ["Cherry"]
            },
            "enable_qwen_fallback": True
        }
    }
    
    try:
        generator = AudioGeneratorFactory.create_generator("forvo_qwen", config["forvo_qwen"])
        
        if forvo_key:
            print("✅ ForvoWithQwenFallback generator created")
        else:
            print("✅ ForvoWithQwenFallback generator created (Qwen-only mode)")
        
        available = AudioGeneratorFactory.get_available_providers(config)
        print(f"✅ Available providers: {available}")
        
        return "forvo_qwen" in available if forvo_key else generator.qwen_generator.is_available()
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Qwen Audio Integration")
    print("Make sure DASHSCOPE_API_KEY is set in your environment")
    print("Optionally set FORVO_API_KEY for full integration testing")
    
    results = []
    
    # Test basic Qwen functionality
    results.append(test_qwen_generator())
    
    # Test factory integration
    results.append(test_factory_integration())
    
    # Test Forvo+Qwen integration
    results.append(test_forvo_qwen_integration())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)