"""
Translation service for multilingual support
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
import httpx
from datetime import datetime
import re

from config.settings import Settings
from utils.logger import ml_logger, log_ml_operation
from utils.cache import MLCacheManager

class TranslationService:
    """
    Service for translating text between supported languages
    """
    
    def __init__(self, settings: Settings, cache_manager: Optional[MLCacheManager] = None):
        """
        Initialize translation service
        
        Args:
            settings: Application settings
            cache_manager: Cache manager instance
        """
        self.settings = settings
        self.cache_manager = cache_manager
        self.supported_languages = settings.supported_languages
        self.default_language = settings.default_language
        self._initialized = False
        
        # Translation mappings for common phrases
        self.phrase_translations = {
            # Transport-related phrases
            "bus": {"rw": "bisi", "fr": "bus"},
            "taxi": {"rw": "takisi", "fr": "taxi"},
            "schedule": {"rw": "gahunda", "fr": "horaire"},
            "fare": {"rw": "ikiguzi", "fr": "tarif"},
            "route": {"rw": "inzira", "fr": "itinéraire"},
            "departure": {"rw": "kugenda", "fr": "départ"},
            "arrival": {"rw": "kugera", "fr": "arrivée"},
            "traffic": {"rw": "ubwikorezi", "fr": "circulation"},
            
            # Common locations
            "kigali": {"rw": "kigali", "fr": "kigali"},
            "huye": {"rw": "huye", "fr": "huye"},
            "musanze": {"rw": "musanze", "fr": "musanze"},
            "rubavu": {"rw": "rubavu", "fr": "rubavu"},
            
            # Time expressions
            "morning": {"rw": "igitondo", "fr": "matin"},
            "afternoon": {"rw": "nyuma ya saa sita", "fr": "après-midi"},
            "evening": {"rw": "nimugoroba", "fr": "soir"},
            "today": {"rw": "uyu munsi", "fr": "aujourd'hui"},
            "tomorrow": {"rw": "ejo", "fr": "demain"},
            "yesterday": {"rw": "ejo hashize", "fr": "hier"},
            
            # Common verbs
            "go": {"rw": "kugenda", "fr": "aller"},
            "come": {"rw": "kuza", "fr": "venir"},
            "help": {"rw": "gufasha", "fr": "aider"},
            "find": {"rw": "gushaka", "fr": "trouver"},
            "know": {"rw": "kumenya", "fr": "savoir"},
            
            # Polite expressions
            "please": {"rw": "nyabuneka", "fr": "s'il vous plaît"},
            "thank you": {"rw": "murakoze", "fr": "merci"},
            "excuse me": {"rw": "mbabarira", "fr": "excusez-moi"},
            "sorry": {"rw": "ihangane", "fr": "désolé"},
        }
    
    async def initialize(self):
        """Initialize the translation service"""
        try:
            start_time = datetime.now()
            ml_logger.info("Initializing translation service...")
            
            # Test external translation service if configured
            if self.settings.google_translate_api_key:
                await self._test_google_translate()
            
            self._initialized = True
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            log_ml_operation("translation_service_init", duration, True, {
                "supported_languages": len(self.supported_languages),
                "phrase_mappings": len(self.phrase_translations),
                "external_service": bool(self.settings.google_translate_api_key)
            })
            
            ml_logger.info("Translation service initialized successfully")
            
        except Exception as e:
            ml_logger.error(f"Failed to initialize translation service: {e}")
            log_ml_operation("translation_service_init", 0, False, {"error": str(e)})
            raise
    
    async def _test_google_translate(self):
        """Test Google Translate API connection"""
        try:
            # Simple test translation
            await self._translate_with_google("hello", "en", "fr")
            ml_logger.info("Google Translate API connection successful")
        except Exception as e:
            ml_logger.warning(f"Google Translate API test failed: {e}")
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        use_cache: bool = True
    ) -> str:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            use_cache: Whether to use caching
            
        Returns:
            Translated text
        """
        if not self._initialized:
            raise ValueError("Translation service not initialized")
        
        if not text or not text.strip():
            return text
        
        # Return original if same language
        if source_lang == target_lang:
            return text
        
        # Validate language codes
        if source_lang not in self.supported_languages or target_lang not in self.supported_languages:
            ml_logger.warning(f"Unsupported language pair: {source_lang} -> {target_lang}")
            return text
        
        try:
            start_time = datetime.now()
            
            # Check cache first
            if use_cache and self.cache_manager:
                cached_translation = await self.cache_manager.get_cached_translation(
                    text, source_lang, target_lang
                )
                if cached_translation:
                    return cached_translation
            
            # Try phrase-based translation first for better accuracy
            phrase_translation = self._translate_phrases(text, source_lang, target_lang)
            if phrase_translation != text:
                translated_text = phrase_translation
                method = "phrase_based"
            else:
                # Use external service or rule-based translation
                if self.settings.google_translate_api_key:
                    translated_text = await self._translate_with_google(text, source_lang, target_lang)
                    method = "google_translate"
                else:
                    translated_text = await self._translate_rule_based(text, source_lang, target_lang)
                    method = "rule_based"
            
            # Cache the result
            if use_cache and self.cache_manager:
                await self.cache_manager.cache_translation(
                    text, source_lang, target_lang, translated_text
                )
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log_ml_operation("translation", duration, True, {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text_length": len(text),
                "method": method
            })
            
            return translated_text
            
        except Exception as e:
            ml_logger.error(f"Translation error: {e}")
            log_ml_operation("translation", 0, False, {
                "error": str(e),
                "source_lang": source_lang,
                "target_lang": target_lang
            })
            return text  # Return original text on error
    
    def _translate_phrases(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using phrase mappings"""
        if source_lang != "en":
            return text  # Phrase mappings are from English
        
        translated_text = text.lower()
        
        # Replace known phrases
        for english_phrase, translations in self.phrase_translations.items():
            if target_lang in translations:
                target_phrase = translations[target_lang]
                
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(english_phrase) + r'\b'
                translated_text = re.sub(pattern, target_phrase, translated_text, flags=re.IGNORECASE)
        
        # Preserve original capitalization pattern
        if text != text.lower():
            translated_text = self._preserve_capitalization(text, translated_text)
        
        return translated_text
    
    async def _translate_with_google(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Google Translate API"""
        if not self.settings.google_translate_api_key:
            raise ValueError("Google Translate API key not configured")
        
        # Map language codes to Google Translate format
        google_lang_map = {
            "rw": "rw",  # Kinyarwanda
            "en": "en",  # English
            "fr": "fr"   # French
        }
        
        google_source = google_lang_map.get(source_lang, source_lang)
        google_target = google_lang_map.get(target_lang, target_lang)
        
        url = "https://translation.googleapis.com/language/translate/v2"
        
        params = {
            "key": self.settings.google_translate_api_key,
            "q": text,
            "source": google_source,
            "target": google_target,
            "format": "text"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, data=params)
            response.raise_for_status()
            
            result = response.json()
            if "data" in result and "translations" in result["data"]:
                return result["data"]["translations"][0]["translatedText"]
            else:
                raise ValueError("Invalid response from Google Translate API")
    
    async def _translate_rule_based(self, text: str, source_lang: str, target_lang: str) -> str:
        """Rule-based translation for basic phrases"""
        
        # Simple rule-based translations for common patterns
        rule_translations = {
            ("en", "rw"): {
                r"\bhello\b": "muraho",
                r"\bhi\b": "muraho",
                r"\bgood morning\b": "mwaramutse",
                r"\bgood afternoon\b": "mwiriwe",
                r"\bgood evening\b": "mwiriwe",
                r"\bthank you\b": "murakoze",
                r"\bthanks\b": "murakoze",
                r"\bplease\b": "nyabuneka",
                r"\bsorry\b": "ihangane",
                r"\bexcuse me\b": "mbabarira",
                r"\bgoodbye\b": "murabeho",
                r"\bbye\b": "murabeho",
                r"\byes\b": "yego",
                r"\bno\b": "oya",
                r"\bwhat\b": "iki",
                r"\bwhen\b": "ryari",
                r"\bwhere\b": "hehe",
                r"\bhow\b": "gute",
                r"\bwhy\b": "kuki"
            },
            ("en", "fr"): {
                r"\bhello\b": "bonjour",
                r"\bhi\b": "salut",
                r"\bgood morning\b": "bonjour",
                r"\bgood afternoon\b": "bon après-midi",
                r"\bgood evening\b": "bonsoir",
                r"\bthank you\b": "merci",
                r"\bthanks\b": "merci",
                r"\bplease\b": "s'il vous plaît",
                r"\bsorry\b": "désolé",
                r"\bexcuse me\b": "excusez-moi",
                r"\bgoodbye\b": "au revoir",
                r"\bbye\b": "salut",
                r"\byes\b": "oui",
                r"\bno\b": "non",
                r"\bwhat\b": "quoi",
                r"\bwhen\b": "quand",
                r"\bwhere\b": "où",
                r"\bhow\b": "comment",
                r"\bwhy\b": "pourquoi"
            },
            ("rw", "en"): {
                r"\bmuraho\b": "hello",
                r"\bmwaramutse\b": "good morning",
                r"\bmwiriwe\b": "good evening",
                r"\bmurakoze\b": "thank you",
                r"\bnyabuneka\b": "please",
                r"\bihangane\b": "sorry",
                r"\bmbabarira\b": "excuse me",
                r"\bmurabeho\b": "goodbye",
                r"\byego\b": "yes",
                r"\boya\b": "no"
            },
            ("fr", "en"): {
                r"\bbonjour\b": "hello",
                r"\bsalut\b": "hi",
                r"\bmerci\b": "thank you",
                r"\bs'il vous plaît\b": "please",
                r"\bdésolé\b": "sorry",
                r"\bexcusez-moi\b": "excuse me",
                r"\bau revoir\b": "goodbye",
                r"\boui\b": "yes",
                r"\bnon\b": "no"
            }
        }
        
        lang_pair = (source_lang, target_lang)
        if lang_pair in rule_translations:
            translated_text = text.lower()
            
            for pattern, replacement in rule_translations[lang_pair].items():
                translated_text = re.sub(pattern, replacement, translated_text, flags=re.IGNORECASE)
            
            # Preserve capitalization
            if text != text.lower():
                translated_text = self._preserve_capitalization(text, translated_text)
            
            return translated_text
        
        return text  # Return original if no rules available
    
    def _preserve_capitalization(self, original: str, translated: str) -> str:
        """Preserve capitalization pattern from original text"""
        if not original or not translated:
            return translated
        
        result = list(translated)
        min_len = min(len(original), len(translated))
        
        for i in range(min_len):
            if original[i].isupper() and result[i].islower():
                result[i] = result[i].upper()
            elif original[i].islower() and result[i].isupper():
                result[i] = result[i].lower()
        
        return ''.join(result)
    
    async def detect_and_translate(
        self,
        text: str,
        target_lang: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[str, str, float]:
        """
        Detect source language and translate to target language
        
        Args:
            text: Text to translate
            target_lang: Target language code
            confidence_threshold: Minimum confidence for translation
            
        Returns:
            Tuple of (translated_text, detected_language, confidence)
        """
        if not text or not text.strip():
            return text, self.default_language, 0.0
        
        try:
            # Detect language (this would integrate with NLP service)
            detected_lang = await self._detect_language_simple(text)
            confidence = 0.8  # Simplified confidence score
            
            if confidence < confidence_threshold:
                return text, detected_lang, confidence
            
            if detected_lang == target_lang:
                return text, detected_lang, confidence
            
            # Translate to target language
            translated_text = await self.translate(text, detected_lang, target_lang)
            
            return translated_text, detected_lang, confidence
            
        except Exception as e:
            ml_logger.error(f"Detect and translate error: {e}")
            return text, self.default_language, 0.0
    
    async def _detect_language_simple(self, text: str) -> str:
        """Simple language detection based on keywords"""
        text_lower = text.lower()
        
        # Kinyarwanda indicators
        rw_indicators = [
            'muraho', 'mwaramutse', 'mwiriwe', 'murakoze', 'nyabuneka', 
            'ihangane', 'mbabarira', 'yego', 'oya', 'bisi', 'gahunda'
        ]
        
        # French indicators
        fr_indicators = [
            'bonjour', 'merci', 'comment', 'pourquoi', 'quand', 'où', 
            'que', 'qui', 'avec', 'dans', 'pour', 'mais'
        ]
        
        # English indicators (default)
        en_indicators = [
            'hello', 'thank', 'how', 'what', 'when', 'where', 'why', 
            'which', 'with', 'from', 'that', 'this'
        ]
        
        # Count indicators
        rw_count = sum(1 for indicator in rw_indicators if indicator in text_lower)
        fr_count = sum(1 for indicator in fr_indicators if indicator in text_lower)
        en_count = sum(1 for indicator in en_indicators if indicator in text_lower)
        
        # Determine language based on highest count
        if rw_count > fr_count and rw_count > en_count:
            return 'rw'
        elif fr_count > en_count:
            return 'fr'
        else:
            return 'en'
    
    async def batch_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        try:
            # Use asyncio.gather for concurrent translation
            tasks = [
                self.translate(text, source_lang, target_lang)
                for text in texts
            ]
            
            translated_texts = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            result = []
            for i, translation in enumerate(translated_texts):
                if isinstance(translation, Exception):
                    ml_logger.warning(f"Translation failed for text {i}: {translation}")
                    result.append(texts[i])  # Return original on error
                else:
                    result.append(translation)
            
            return result
            
        except Exception as e:
            ml_logger.error(f"Batch translation error: {e}")
            return texts  # Return original texts on error
    
    async def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages with their names"""
        return {
            "en": "English",
            "rw": "Kinyarwanda",
            "fr": "Français"
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the translation service"""
        return {
            "initialized": self._initialized,
            "supported_languages": self.supported_languages,
            "default_language": self.default_language,
            "phrase_mappings": len(self.phrase_translations),
            "external_service": {
                "google_translate": bool(self.settings.google_translate_api_key),
                "service_type": self.settings.translation_service
            },
            "cache_enabled": bool(self.cache_manager)
        }
    
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized