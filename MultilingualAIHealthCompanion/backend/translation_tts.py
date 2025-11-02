import asyncio
import httpx
from typing import Dict, Any
import os
import logging
import tempfile
import uuid
import wave
import array
import math
import base64

# Set up logging for this module
logger = logging.getLogger(__name__)

# Define the audio files directory
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)  # Create directory if it doesn't exist


async def generate_translations_and_tts(summaries: Dict[str, Any], source_language: str = "en") -> Dict[str, str]:
    """
    Generate translations of the summaries into Hindi and Marathi,
    and create audio files using ElevenLabs TTS with fallback to local TTS.
    """
    logger.info("Starting translation and TTS generation")
    translated_content = {}
    
    # Define the target languages
    target_languages = {
        "en": "English",
        "hi": "Hindi", 
        "mr": "Marathi"
    }
    
    # Get the text to translate (for now, use the plain language summary)
    text_to_translate = summaries["plain_language"]
    logger.debug(f"Text to translate length: {len(text_to_translate)}")
    
    for lang_code, lang_name in target_languages.items():
        logger.debug(f"Processing language: {lang_code}")
        if source_language != lang_code:
            # Translate the text
            translated_text = translate_text(text_to_translate, source_language, lang_code)
            
            # Generate TTS audio for the translated text
            audio_url = await generate_tts_audio(translated_text, lang_code)
            
            translated_content[lang_code] = {
                "text": translated_text,
                "audio_url": audio_url
            }
        else:
            # If the source language is the same as target, use original text
            # This will now also generate audio for English if it's the source language
            audio_url = await generate_tts_audio(text_to_translate, lang_code)
            translated_content[lang_code] = {
                "text": text_to_translate,
                "audio_url": audio_url
            }
    
    logger.info("Translation and TTS generation completed")
    return translated_content


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source language to target language.
    This is a placeholder that would integrate with a translation API in production.
    """
    logger.debug(f"Translating from {source_lang} to {target_lang}")
    # In a real implementation, this would call a translation API like Google Translate
    # For now, return a placeholder
    
    # Map language codes to names for the placeholder
    lang_names = {
        "en": "English", 
        "hi": "Hindi", 
        "mr": "Marathi"
    }
    
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    result = f"[This would be the {target_name} translation of: {text[:100]}...]"  # Truncate for placeholder
    logger.debug(f"Translation completed: {len(result)} characters")
    return result


async def generate_tts_audio(text: str, language_code: str) -> str:
    """
    Generate audio file from text using ElevenLabs TTS service with local fallback.
    """
    logger.debug(f"Generating TTS audio for language: {language_code}")
    
    # First try ElevenLabs API
    elevenlabs_result = await generate_tts_audio_elevenlabs(text, language_code)
    
    # If ElevenLabs fails or doesn't have an API key, fall back to local TTS
    if elevenlabs_result and not elevenlabs_result.startswith("http://example.com/audio/"):
        logger.info(f"Successfully generated TTS audio via ElevenLabs for {language_code}")
        return elevenlabs_result
    else:
        logger.warning(f"ElevenLabs failed for {language_code}, falling back to local TTS")
        return await generate_tts_audio_local(text, language_code)


async def generate_tts_audio_elevenlabs(text: str, language_code: str) -> str:
    """
    Generate audio file from text using ElevenLabs TTS service.
    """
    logger.debug(f"Attempting TTS via ElevenLabs for language: {language_code}")
    
    # Get ElevenLabs API key from environment
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.debug("ELEVENLABS_API_KEY not found, skipping ElevenLabs")
        return None
    
    # Map language codes to appropriate ElevenLabs voice IDs
    # Using some popular voice IDs as examples
    voice_mapping = {
        "hi": "Zjj2iX3aHYDcJSG4mMzk",  # This would be an actual Hindi voice ID in production
        "mr": "Zjj2iX3aHYDcJSG4mMzk",  # This would be an actual Marathi voice ID in production
        "en": "Zjj2iX3aHYDcJSG4mMzk"   # This would be an actual English voice ID in production
    }
    
    # Use a default voice ID if language is not in mapping
    # In a production system, you would select actual voice IDs based on language
    voice_id = voice_mapping.get(language_code, "Zjj2iX3aHYDcJSG4mMzk")
    
    # ElevenLabs API endpoint
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    # Headers for the API request
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Payload for the API request
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        # Make the API request using httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                # Save the audio content to a file in the audio_files directory
                filename = f"tts_{uuid.uuid4()}.mp3"
                audio_path = os.path.join(AUDIO_DIR, filename)
                
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                    f.flush()  # Ensure data is written
                
                # Return the path to access the audio file via the API endpoint
                audio_url = f"/audio/{filename}"
                logger.info(f"Successfully generated TTS audio via ElevenLabs for {language_code}")
                return audio_url
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                # Return placeholder if API call fails
                error_audio_id = str(uuid.uuid4())
                return f"http://example.com/audio/{error_audio_id}.mp3"
                
    except Exception as e:
        logger.error(f"Error during ElevenLabs TTS generation: {str(e)}")
        # Return placeholder if any exception occurs
        exception_audio_id = str(uuid.uuid4())
        return f"http://example.com/audio/{exception_audio_id}.mp3"


async def generate_tts_audio_local(text: str, language_code: str) -> str:
    """
    Generate audio file from text using local TTS.
    This creates a simple waveform-based audio as a placeholder.
    In production, you might use pyttsx3 or other local TTS libraries.
    """
    logger.debug(f"Generating local TTS audio for language: {language_code}")
    
    try:
        # Create a simple placeholder audio file since we can't use pyttsx3 in async context
        # This creates a basic WAV file as a placeholder
        temp_audio_path = create_placeholder_audio(text, language_code)
        
        # Move the temporary file to the audio_files directory with a proper name
        filename = f"tts_local_{uuid.uuid4()}.wav"
        audio_path = os.path.join(AUDIO_DIR, filename)
        
        # Move the temporary file to the audio_files directory
        import shutil
        shutil.move(temp_audio_path, audio_path)
        
        # Return the path to access the audio file via the API endpoint
        audio_url = f"/audio/{filename}"
        return audio_url
    except Exception as e:
        logger.error(f"Error during local TTS generation: {str(e)}")
        # Return fallback placeholder if local TTS fails
        audio_id = str(uuid.uuid4())
        return f"http://example.com/audio/{audio_id}.mp3"


def create_placeholder_audio(text: str, language_code: str) -> str:
    """
    Create a placeholder audio file with simple waveform data.
    This is just a placeholder since actual TTS would require more complex processing.
    """
    try:
        import tempfile
        import wave
        import math
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name
            
            # Create simple audio (a basic tone pattern as placeholder)
            sample_rate = 22050  # CD quality
            duration = min(len(text) / 10, 10)  # Duration based on text length, max 10 seconds
            frames = int(sample_rate * duration)
            
            # Generate simple waveform
            wave_data = []
            for i in range(frames):
                # Create a combination of frequencies as a placeholder
                sample = int(32767.0 * (0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate) +
                                       0.3 * math.sin(2.0 * math.pi * 554.0 * i / sample_rate) +
                                       0.2 * math.sin(2.0 * math.pi * 660.0 * i / sample_rate)))
                wave_data.append(sample)
            
            # Write the waveform to a WAV file
            with wave.open(temp_filename, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(array.array('h', wave_data).tobytes())
            
            return temp_filename
    except Exception as e:
        logger.error(f"Error creating placeholder audio: {str(e)}")
        # Return placeholder if file creation fails
        audio_id = str(uuid.uuid4())
        temp_filename = f"temp_audio_{audio_id}.mp3"
        return temp_filename


def get_elevenlabs_voices(language_code: str):
    """
    Placeholder function to get appropriate voices for the target language.
    """
    logger.debug(f"Getting ElevenLabs voices for language: {language_code}")
    # In a real implementation, this would fetch voice options from ElevenLabs
    voice_mapping = {
        "hi": "hi-IN-MadhurNeural",  # Example voice for Hindi
        "mr": "mr-IN-AarohiNeural",  # Example voice for Marathi
        "en": "en-US-JennyNeural"    # Example voice for English
    }
    
    result = voice_mapping.get(language_code, "en-US-JennyNeural")
    logger.debug(f"Selected voice: {result}")
    return result