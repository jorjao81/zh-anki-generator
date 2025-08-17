import os
import sys
import requests
import base64
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tts.v20190823 import tts_client, models
from pydub import AudioSegment
import io
import random

if len(sys.argv) != 2:
    print("Usage: python test_tencent.py <word>")
    sys.exit(1)

text = sys.argv[1]

# Tencent TTS voices for Chinese (Mandarin)
voices = [
    101052,  # zhiwei (Chinese)
    101053,  # zhifang (Chinese)
    101054,  # zhiyou (Chinese)
    101001,  # zhiyu (Chinese)
    101002   # zhiling (Chinese)
]

selected_voice = random.choice(voices)
voice_names = {
    101052: "zhiwei (Chinese)",
    101053: "zhifang (Chinese)", 
    101054: "zhiyou (Chinese)",
    101001: "zhiyu (Chinese)",
    101002: "zhiling (Chinese)"
}

print(f"Using voice: {voice_names.get(selected_voice, selected_voice)}")

# Get credentials from environment
secret_id = os.getenv("TENCENT_SECRET_ID")
secret_key = os.getenv("TENCENT_API_KEY")

if not secret_id or not secret_key:
    print("Error: TENCENT_SECRET_ID and TENCENT_API_KEY environment variables must be set")
    sys.exit(1)

try:
    # Create credential object
    cred = credential.Credential(secret_id, secret_key)
    
    # Configure client
    httpProfile = HttpProfile()
    httpProfile.endpoint = "tts.tencentcloudapi.com"
    
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    
    # Create TTS client - try Singapore region
    client = tts_client.TtsClient(cred, "ap-singapore", clientProfile)
    
    # Create request
    req = models.TextToVoiceRequest()
    req.Text = text
    req.VoiceType = selected_voice
    req.Codec = "mp3"  # Get MP3 format directly
    req.SampleRate = 16000
    req.SessionId = f"tts_session_{random.randint(100000, 999999)}"  # Generate unique session ID
    
    # Call API
    print("Calling Tencent TTS API...")
    resp = client.TextToVoice(req)
    print("TTS request successful")
    
    # Get audio data (base64 encoded)
    audio_base64 = resp.Audio
    audio_data = base64.b64decode(audio_base64)
    
    # Save MP3 data directly (no conversion needed)
    save_path = "downloaded_audio_tencent.mp3"
    with open(save_path, 'wb') as f:
        f.write(audio_data)
    
    print(f"Audio file saved to: {save_path}")
    
    # Play the audio automatically using Python library
    try:
        from playsound3 import playsound
        print("🔊 Playing audio...")
        playsound(save_path)
        print("✅ Audio played successfully")
    except ImportError:
        print("❌ playsound3 not available - install with: pip install playsound3")
    except Exception as play_error:
        print(f"❌ Could not play audio: {play_error}")
    
except Exception as e:
    print(f"TTS generation failed: {str(e)}")
