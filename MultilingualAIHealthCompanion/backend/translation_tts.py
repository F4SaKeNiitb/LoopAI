import asyncio
import httpx
from typing import Dict, Any
import os
import logging

# Set up logging for this module
logger = logging.getLogger(__name__)


async def generate_translations_and_tts(summaries: Dict[str, Any], source_language: str = "en") -> Dict[str, str]:
    """
    Generate translations of the summaries into Hindi and Marathi,
    and create audio files using ElevenLabs TTS.
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
    Generate audio file from text using ElevenLabs TTS service.
    """
    logger.debug(f"Generating TTS audio for language: {language_code}")
    
    # Get ElevenLabs API key from environment
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not found, returning placeholder audio")
        # Generate a placeholder audio URL
        import uuid
        audio_id = str(uuid.uuid4())
        return f"http://example.com/audio/{audio_id}.mp3"
    
    # Map language codes to appropriate ElevenLabs voice IDs
    # Using some popular voice IDs as examples
    voice_mapping = {
        "hi": "Zjj2iX3aHYDcJSG4mMzk",  # This would be an actual Hindi voice ID in production
        "mr": "Zjj2iX3aHYDcJSG4mMzk",  # This would be an actual Marathi voice ID in production
        "en": "Zjj2iX3aHYDcJSG4mMzk"   # This would be an actual English voice ID in production
    }
    
    # Use a default voice ID if language is not in mapping
    # In a production system, you would select actual voice IDs based on language
    voice_id = voice_mapping.get(language_code)
    
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
                # In a real system, we would save the audio content to a file
                # and return a URL to access it. For now, we'll just return a placeholder.
                # The actual audio content is in response.content
                import uuid
                filename = f"tts_{uuid.uuid4()}.mp3"
                
                # In a production system, you would do something like:
                # with open(f"audio_files/{filename}", "wb") as f:
                #     f.write(response.content)
                #     f.flush()  # Ensure data is written
                
                # For now, return a direct URL from ElevenLabs or use the API response directly
                # Since we don't have actual files stored locally, return an example URL
                # that could be handled by a client-side player or return a placeholder
                audio_url = f"http://example.com/audio/{filename}"
                logger.info(f"Successfully generated TTS audio for {language_code}")
                return audio_url
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                # Return placeholder if API call fails
                import uuid
                audio_id = str(uuid.uuid4())
                return f"http://example.com/audio/{audio_id}.mp3"
                
    except Exception as e:
        logger.error(f"Error during ElevenLabs TTS generation: {str(e)}")
        # Return placeholder if any exception occurs
        import uuid
        audio_id = str(uuid.uuid4())
        return f"http://example.com/audio/{audio_id}.mp3"





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