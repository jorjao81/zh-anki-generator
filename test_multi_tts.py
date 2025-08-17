#!/usr/bin/env python3
"""Test script for Multi-TTS integration."""

import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add src to path
import sys
sys.path.insert(0, 'src')

from anki_pleco_importer.audio import AudioGeneratorFactory

def test_multi_tts_provider():
    """Test forvo_multi_tts provider."""
    print("=" * 50)
    print("Testing Forvo Multi-TTS Provider")
    print("=" * 50)
    
    # Test config with all providers
    config = {
        "forvo_multi_tts": {
            "forvo": {
                "api_key": os.environ.get('FORVO_API_KEY'),
                "preferred_users": ["nonexistent_user"],  # Force fallback to show options
                "interactive_selection": True
            },
            "qwen": {
                "api_key": os.environ.get('DASHSCOPE_API_KEY'),
                "voices": ["Cherry", "Chelsie"]  # Subset for speed
            },
            "tencent": {
                "secret_id": os.environ.get('TENCENT_SECRET_ID'),
                "secret_key": os.environ.get('TENCENT_API_KEY'),
                "voices": [101052, 101001],  # Subset for speed
                "region": "ap-singapore"
            },
            "enable_tts_fallback": True
        }
    }
    
    try:
        # Test creating multi-TTS generator through factory
        generator = AudioGeneratorFactory.create_generator("forvo_multi_tts", config["forvo_multi_tts"])
        print("✅ Factory can create Forvo Multi-TTS generator")
        
        # Test getting available providers
        available = AudioGeneratorFactory.get_available_providers(config)
        print(f"✅ Available providers: {available}")
        
        if "forvo_multi_tts" in available:
            print("✅ Forvo Multi-TTS provider is available")
            return True
        else:
            print("❌ Forvo Multi-TTS provider not available")
            return False
            
    except Exception as e:
        print(f"❌ Multi-TTS test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Multi-TTS Audio Integration")
    print("This will show Forvo + Qwen + Tencent voices all together")
    
    result = test_multi_tts_provider()
    
    if result:
        print("\n🎉 Multi-TTS provider is ready!")
        print("Use: --audio-providers forvo_multi_tts")
    else:
        print("\n❌ Multi-TTS provider setup failed")