"""
Core NLP service for language processing and model management
"""

import spacy
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import re
from datetime import datetime

from config.settings import Settings
from utils.logger import ml_logger, log_ml_operation

class NLPService:
    """
    Core NLP service for text processing and language operations
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize NLP service
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.models: Dict[str, Any] = {}
        self.language_models: Dict[str, str] = {
            'en': settings.spacy_model_en,
            'fr': settings.spacy_model_fr
        }
        self._initialized = False
    
    async def initialize(self):
        """Initialize NLP models and resources"""
        try:
            start_time = datetime.now()
            ml_logger.info("Initializing NLP service...")
            
            # Load spaCy models
            await self._load_spacy_models()
            
            # Initialize text processors
            self._initialize_text_processors()
            
            self._initialized = True
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            log_ml_operation("nlp_service_init", duration, True, {
                "models_loaded": len(self.models),
                "languages": list(self.language_models.keys())
            })
            
            ml_logger.info("NLP service initialized successfully")
            
        except Exception as e:
            ml_logger.error(f"Failed to initialize NLP service: {e}")
            log_ml_operation("nlp_service_init", 0, False, {"error": str(e)})
            raise
    
    async def _load_spacy_models(self):
        """Load spaCy models for different languages"""
        for lang, model_name in self.language_models.items():
            try:
                ml_logger.info(f"Loading spaCy model for {lang}: {model_name}")
                
                # Load model in thread pool to avoid blocking
                nlp = await asyncio.get_event_loop().run_in_executor(
                    None, spacy.load, model_name
                )
                
                # Configure pipeline
                if 'ner' not in nlp.pipe_names:
                    nlp.add_pipe('ner')
                
                self.models[f'spacy_{lang}'] = nlp
                ml_logger.info(f"Successfully loaded spaCy model for {lang}")
                
            except Exception as e:
                ml_logger.warning(f"Failed to load spaCy model for {lang}: {e}")
                # Continue without this model
                continue
    
    def _initialize_text_processors(self):
        """Initialize text preprocessing utilities"""
        # Regex patterns for text cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Stop words for different languages
        self.stop_words = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
            'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'dans', 'sur', 'à', 'pour', 'de', 'avec', 'par'},
            'rw': {'ni', 'na', 'no', 'mu', 'ku', 'muri', 'kuri', 'cya', 'bya'}
        }
    
    async def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        if not text or not text.strip():
            return self.settings.default_language
        
        try:
            start_time = datetime.now()
            
            # Clean text for better detection
            cleaned_text = self._clean_text_for_detection(text)
            
            if len(cleaned_text) < 3:
                return self.settings.default_language
            
            # Use langdetect for primary detection
            detected_lang = detect(cleaned_text)
            
            # Map to supported languages
            if detected_lang in self.settings.supported_languages:
                result_lang = detected_lang
            elif detected_lang in ['rw', 'kin']:  # Kinyarwanda variants
                result_lang = 'rw'
            else:
                # Fallback: check for language-specific keywords
                result_lang = self._detect_by_keywords(cleaned_text)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log_ml_operation("language_detection", duration, True, {
                "detected": result_lang,
                "text_length": len(text)
            })
            
            return result_lang
            
        except LangDetectException:
            # Fallback to keyword-based detection
            return self._detect_by_keywords(text)
        except Exception as e:
            ml_logger.error(f"Language detection error: {e}")
            return self.settings.default_language
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        text = self.phone_pattern.sub('', text)
        
        # Remove excessive whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove punctuation but keep letters and spaces
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
        
        return text.strip().lower()
    
    def _detect_by_keywords(self, text: str) -> str:
        """Fallback language detection using keywords"""
        text_lower = text.lower()
        
        # Kinyarwanda keywords
        rw_keywords = ['muraho', 'mwaramutse', 'amakuru', 'gusa', 'cyangwa', 'ariko', 'kandi', 'niba']
        rw_score = sum(1 for keyword in rw_keywords if keyword in text_lower)
        
        # French keywords
        fr_keywords = ['bonjour', 'merci', 'comment', 'pourquoi', 'quand', 'où', 'que', 'qui']
        fr_score = sum(1 for keyword in fr_keywords if keyword in text_lower)
        
        # English keywords
        en_keywords = ['hello', 'thank', 'how', 'what', 'when', 'where', 'why', 'which']
        en_score = sum(1 for keyword in en_keywords if keyword in text_lower)
        
        # Return language with highest score
        scores = {'rw': rw_score, 'fr': fr_score, 'en': en_score}
        detected = max(scores, key=scores.get)
        
        return detected if scores[detected] > 0 else self.settings.default_language
    
    async def preprocess_text(
        self,
        text: str,
        language: Optional[str] = None,
        remove_stop_words: bool = False,
        lemmatize: bool = False
    ) -> str:
        """
        Preprocess text for NLP tasks
        
        Args:
            text: Input text
            language: Text language (auto-detected if None)
            remove_stop_words: Whether to remove stop words
            lemmatize: Whether to lemmatize tokens
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        try:
            start_time = datetime.now()
            
            if not language:
                language = await self.detect_language(text)
            
            # Basic cleaning
            processed_text = self._basic_text_cleaning(text)
            
            # Use spaCy for advanced preprocessing if available
            if f'spacy_{language}' in self.models:
                processed_text = await self._spacy_preprocess(
                    processed_text, language, remove_stop_words, lemmatize
                )
            elif remove_stop_words:
                processed_text = self._remove_stop_words(processed_text, language)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log_ml_operation("text_preprocessing", duration, True, {
                "language": language,
                "input_length": len(text),
                "output_length": len(processed_text)
            })
            
            return processed_text.strip()
            
        except Exception as e:
            ml_logger.error(f"Text preprocessing error: {e}")
            return self._basic_text_cleaning(text)
    
    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning operations"""
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def _spacy_preprocess(
        self,
        text: str,
        language: str,
        remove_stop_words: bool,
        lemmatize: bool
    ) -> str:
        """Use spaCy for advanced text preprocessing"""
        nlp = self.models[f'spacy_{language}']
        
        # Process text in thread pool
        doc = await asyncio.get_event_loop().run_in_executor(
            None, nlp, text
        )
        
        tokens = []
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
            
            # Skip stop words if requested
            if remove_stop_words and token.is_stop:
                continue
            
            # Use lemma if requested, otherwise use original text
            token_text = token.lemma_ if lemmatize else token.text
            tokens.append(token_text.lower())
        
        return ' '.join(tokens)
    
    def _remove_stop_words(self, text: str, language: str) -> str:
        """Remove stop words for given language"""
        if language not in self.stop_words:
            return text
        
        words = text.lower().split()
        filtered_words = [word for word in words if word not in self.stop_words[language]]
        return ' '.join(filtered_words)
    
    async def extract_keywords(
        self,
        text: str,
        language: Optional[str] = None,
        max_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            language: Text language
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keyword dictionaries with text and score
        """
        try:
            if not language:
                language = await self.detect_language(text)
            
            start_time = datetime.now()
            keywords = []
            
            # Use spaCy if available
            if f'spacy_{language}' in self.models:
                keywords = await self._spacy_extract_keywords(text, language, max_keywords)
            else:
                # Simple frequency-based keyword extraction
                keywords = self._simple_keyword_extraction(text, language, max_keywords)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log_ml_operation("keyword_extraction", duration, True, {
                "language": language,
                "keywords_found": len(keywords)
            })
            
            return keywords
            
        except Exception as e:
            ml_logger.error(f"Keyword extraction error: {e}")
            return []
    
    async def _spacy_extract_keywords(
        self,
        text: str,
        language: str,
        max_keywords: int
    ) -> List[Dict[str, Any]]:
        """Extract keywords using spaCy"""
        nlp = self.models[f'spacy_{language}']
        
        # Process text
        doc = await asyncio.get_event_loop().run_in_executor(None, nlp, text)
        
        # Extract noun phrases and named entities
        keywords = {}
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit phrase length
                key = chunk.text.lower().strip()
                if key and len(key) > 2:
                    keywords[key] = keywords.get(key, 0) + 1
        
        # Add named entities
        for ent in doc.ents:
            key = ent.text.lower().strip()
            if key and len(key) > 2:
                keywords[key] = keywords.get(key, 0) + 2  # Higher weight for entities
        
        # Add important single tokens
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                key = token.lemma_.lower()
                keywords[key] = keywords.get(key, 0) + 0.5
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"text": keyword, "score": score}
            for keyword, score in sorted_keywords[:max_keywords]
        ]
    
    def _simple_keyword_extraction(
        self,
        text: str,
        language: str,
        max_keywords: int
    ) -> List[Dict[str, Any]]:
        """Simple frequency-based keyword extraction"""
        # Clean and tokenize
        cleaned_text = self._clean_text_for_detection(text)
        words = cleaned_text.split()
        
        # Remove stop words
        if language in self.stop_words:
            words = [word for word in words if word not in self.stop_words[language]]
        
        # Count frequencies
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"text": word, "score": freq}
            for word, freq in sorted_words[:max_keywords]
        ]
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "initialized": self._initialized,
            "models_loaded": list(self.models.keys()),
            "supported_languages": self.settings.supported_languages,
            "default_language": self.settings.default_language,
            "spacy_models": {
                lang: model_name 
                for lang, model_name in self.language_models.items()
                if f'spacy_{lang}' in self.models
            }
        }
    
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized