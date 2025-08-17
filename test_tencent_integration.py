#!/usr/bin/env python3
"""Test script for Tencent integration with audio generation."""

import os
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add src to path
import sys
sys.path.insert(0, 'src')

from anki_pleco_importer.audio import TencentGenerator, AudioGeneratorFactory

def test_tencent_generator():
    """Test Tencent generator basic functionality."""
    print("=" * 50)
    print("Testing Tencent Generator")
    print("=" * 50)
    
    # Check for API credentials
    secret_id = os.environ.get('TENCENT_SECRET_ID')
    secret_key = os.environ.get('TENCENT_API_KEY')
    if not secret_id or not secret_key:
        print("⚠️  TENCENT_SECRET_ID and TENCENT_API_KEY environment variables not set")
        print("   Skipping Tencent tests")
        return False
    
    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test TencentGenerator
        config = {
            "secret_id": secret_id,
            "secret_key": secret_key,
            "voices": [101052, 101001]  # Test with subset for speed
        }
        
        generator = TencentGenerator(
            secret_id=config["secret_id"],
            secret_key=config["secret_key"],
            cache_dir=temp_dir,
            voices=config["voices"]
        )
        
        if not generator.is_available():
            print("❌ Tencent generator not available")
            return False
        
        print("✅ Tencent generator is available")
        
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
    
    # Test tencent provider
    config = {
        "tencent": {
            "secret_id": os.environ.get('TENCENT_SECRET_ID'),
            "secret_key": os.environ.get('TENCENT_API_KEY'),
            "voices": [101052]
        }
    }
    
    if not config["tencent"]["secret_id"] or not config["tencent"]["secret_key"]:
        print("⚠️  TENCENT credentials not set, skipping factory test")
        return False
    
    try:
        # Test creating tencent generator through factory
        generator = AudioGeneratorFactory.create_generator("tencent", config["tencent"])
        print("✅ Factory can create Tencent generator")
        
        # Test getting available providers
        available = AudioGeneratorFactory.get_available_providers(config)
        print(f"✅ Available providers: {available}")
        
        return "tencent" in available
        
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Tencent Audio Integration")
    print("Make sure TENCENT_SECRET_ID and TENCENT_API_KEY are set in your environment")
    
    results = []
    
    # Test basic Tencent functionality
    results.append(test_tencent_generator())
    
    # Test factory integration
    results.append(test_factory_integration())
    
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