"""Audio enhancement utilities using Facebook Denoiser for speech quality improvement."""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional
import warnings

logger = logging.getLogger(__name__)

# Suppress warnings from denoiser and torch
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioEnhancer:
    """Audio enhancement using Facebook Denoiser for speech denoising."""
    
    def __init__(self):
        self._denoiser_available = False
        self._denoiser = None
        self._device = None
        self._initialize_denoiser()
    
    def _initialize_denoiser(self):
        """Initialize the Facebook Denoiser model."""
        try:
            import torch
            from denoiser import pretrained
            
            # Check if CUDA is available, but default to CPU for compatibility
            self._device = torch.device("cpu")  # Always use CPU for reliability
            
            logger.info("Loading Facebook Denoiser model (dns64)...")
            # Use the dns64 model which is the best quality according to research
            self._denoiser = pretrained.dns64().to(self._device)
            self._denoiser.eval()
            self._denoiser_available = True
            
            logger.info("✅ Facebook Denoiser initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Facebook Denoiser not available: {e}")
            logger.warning("Audio will use normalization only. Install with: pip install denoiser torch")
            self._denoiser_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Facebook Denoiser: {e}")
            logger.warning("Audio will use normalization only")
            self._denoiser_available = False
    
    def is_available(self) -> bool:
        """Check if the denoiser is available."""
        return self._denoiser_available
    
    def denoise_audio_file(self, input_path: str, output_path: str) -> bool:
        """
        Denoise audio file using Facebook Denoiser.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output denoised audio file
            
        Returns:
            True if successful, False if failed
        """
        if not self._denoiser_available:
            logger.debug("Denoiser not available, skipping denoising")
            return False
            
        try:
            import torch
            import torchaudio
            import soundfile as sf
            
            logger.debug(f"Denoising audio: {input_path} -> {output_path}")
            
            # Load audio file
            try:
                # Try with librosa first for better MP3 support
                import librosa
                audio_data, sample_rate = librosa.load(input_path, sr=None, mono=True)
                # Convert to torch tensor
                audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).to(self._device)
            except ImportError:
                # Fallback to torchaudio
                audio_tensor, sample_rate = torchaudio.load(input_path)
                audio_tensor = audio_tensor.to(self._device)
                
                # Ensure mono
                if audio_tensor.shape[0] > 1:
                    audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed (Denoiser expects 16kHz)
            target_sr = 16000
            if sample_rate != target_sr:
                try:
                    import librosa
                    audio_np = audio_tensor.cpu().numpy().squeeze()
                    audio_resampled = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=target_sr)
                    audio_tensor = torch.FloatTensor(audio_resampled).unsqueeze(0).to(self._device)
                    sample_rate = target_sr
                except ImportError:
                    # Use torchaudio resampling as fallback
                    resampler = torchaudio.transforms.Resample(sample_rate, target_sr).to(self._device)
                    audio_tensor = resampler(audio_tensor)
                    sample_rate = target_sr
            
            # Apply denoiser
            with torch.no_grad():
                # Add batch dimension if needed
                if len(audio_tensor.shape) == 2:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                enhanced = self._denoiser(audio_tensor)
                
                # Remove batch dimension
                enhanced = enhanced.squeeze(0)
            
            # Convert back to numpy
            enhanced_audio = enhanced.cpu().numpy()
            
            # Ensure mono output
            if len(enhanced_audio.shape) > 1:
                enhanced_audio = enhanced_audio[0]
            
            # Save enhanced audio
            sf.write(output_path, enhanced_audio, sample_rate, format='mp3')
            
            logger.debug(f"✅ Successfully denoised audio: {output_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to denoise audio {input_path}: {e}")
            return False
    
    def enhance_audio_file(self, input_path: str, output_path: str, target_dbfs: float = -15.0) -> bool:
        """
        Enhance audio file with denoising and normalization.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output enhanced audio file
            target_dbfs: Target volume level in dBFS
            
        Returns:
            True if successful, False if failed
        """
        try:
            # Step 1: Denoise if available
            if self._denoiser_available:
                # Create temporary file for denoised audio
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    temp_denoised = temp_file.name
                
                try:
                    if self.denoise_audio_file(input_path, temp_denoised):
                        logger.debug("Denoising successful, proceeding with normalization")
                        input_for_normalization = temp_denoised
                    else:
                        logger.debug("Denoising failed, using original file for normalization")
                        input_for_normalization = input_path
                finally:
                    # Clean up temp file later
                    pass
            else:
                logger.debug("Denoiser not available, using original file for normalization")
                input_for_normalization = input_path
            
            # Step 2: Normalize volume
            self._normalize_audio_volume(input_for_normalization, output_path, target_dbfs)
            
            # Clean up temporary file if it was created
            if self._denoiser_available and 'temp_denoised' in locals():
                try:
                    os.unlink(temp_denoised)
                except Exception:
                    pass
            
            logger.debug(f"✅ Successfully enhanced audio: {output_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to enhance audio {input_path}: {e}")
            return False
    
    def _normalize_audio_volume(self, input_path: str, output_path: str, target_dbfs: float) -> None:
        """Normalize audio volume to target dBFS level."""
        try:
            from pydub import AudioSegment
            
            logger.debug(f"Normalizing audio volume: {input_path} to {target_dbfs} dBFS")
            
            # Load the audio file
            audio = AudioSegment.from_file(input_path)
            
            # Calculate the gain adjustment needed
            current_dbfs = audio.dBFS
            gain_adjustment = target_dbfs - current_dbfs
            
            logger.debug(f"Current volume: {current_dbfs:.2f} dBFS, adjustment: {gain_adjustment:.2f} dB")
            
            # Apply gain adjustment if needed
            if abs(gain_adjustment) > 0.1:  # Only adjust if difference is significant
                normalized_audio = audio.apply_gain(gain_adjustment)
                
                # Export to output file
                normalized_audio.export(output_path, format="mp3")
                logger.debug(f"Normalized audio volume: {current_dbfs:.2f} → {target_dbfs:.2f} dBFS")
            else:
                logger.debug(f"Audio already at target volume ({current_dbfs:.2f} dBFS), copying file")
                # Just copy the file if no adjustment needed
                import shutil
                shutil.copy2(input_path, output_path)
                
        except Exception as e:
            logger.warning(f"Failed to normalize audio volume for {input_path}: {e}")
            # Copy original file as fallback
            try:
                import shutil
                shutil.copy2(input_path, output_path)
            except Exception:
                pass


# Global instance for reuse
_audio_enhancer = None

def get_audio_enhancer() -> AudioEnhancer:
    """Get the global AudioEnhancer instance."""
    global _audio_enhancer
    if _audio_enhancer is None:
        _audio_enhancer = AudioEnhancer()
    return _audio_enhancer