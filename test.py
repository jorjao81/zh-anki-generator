import os
import sys
import requests
import dashscope
from pydub import AudioSegment
import io
import random

if len(sys.argv) != 2:
    print("Usage: python test.py <word>")
    sys.exit(1)

text = sys.argv[1]
#voices = ["Chelsie", "Serena"]
voices = ["Chelsie", "Cherry", "Ethan", "Serena"]
selected_voice = random.choice(voices)

print(f"Using voice: {selected_voice}")

response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
    model="qwen-tts",
    text=text,
    voice=selected_voice,
)
print(response)
audio_url = response.output.audio["url"]
save_path = "downloaded_audio.mp3"  # Save as MP3

try:
    response = requests.get(audio_url)
    response.raise_for_status()  # Check if the request was successful
    
    # Convert WAV data to MP3 in memory
    audio_data = io.BytesIO(response.content)
    audio = AudioSegment.from_wav(audio_data)
    audio.export(save_path, format="mp3")
    
    print(f"Audio file saved to: {save_path}")
except Exception as e:
    print(f"Download failed: {str(e)}")
