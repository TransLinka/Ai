"""
Configuration settings for the ML service
"""

from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Basic configuration
    app_name: str = "Translinka ML Service"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 5000
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 300  # 5 minutes default
    
    # Database configuration (for analytics and model storage)
    database_url: Optional[str] = None
    
    # Model paths and configuration
    models_dir: str = "./models"
    
    # NLP Models
    spacy_model_en: str = "en_core_web_sm"
    spacy_model_fr: str = "fr_core_news_sm"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    # Intent Classification
    intent_model_path: str = "./models/intent_classifier.joblib"
    intent_vectorizer_path: str = "./models/intent_vectorizer.joblib"
    intent_confidence_threshold: float = 0.6
    
    # Entity Extraction
    entity_model_path: str = "./models/entity_extractor.joblib"
    custom_entities_path: str = "./data/custom_entities.json"
    
    # Language Detection and Translation
    translation_service: str = "google"  # google, azure, aws, local
    google_translate_api_key: Optional[str] = None
    translation_cache_ttl: int = 3600  # 1 hour
    
    # Response Generation
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 150
    response_templates_path: str = "./data/response_templates.json"
    
    # Supported languages
    supported_languages: List[str] = ["en", "rw", "fr"]
    default_language: str = "en"
    
    # Intent categories and confidence thresholds
    intent_categories: Dict[str, float] = {
        "transport_query": 0.7,
        "entertainment_request": 0.6,
        "support_request": 0.6,
        "greeting": 0.5,
        "goodbye": 0.5,
        "general": 0.4
    }
    
    # Entity types
    entity_types: List[str] = [
        "LOCATION",
        "TIME",
        "DATE",
        "PERSON",
        "TRANSPORT_TYPE",
        "ROUTE",
        "FARE",
        "ENTERTAINMENT_TYPE"
    ]
    
    # Caching configuration
    cache_config: Dict[str, int] = {
        "intent_classification": 300,    # 5 minutes
        "entity_extraction": 300,        # 5 minutes
        "translation": 3600,             # 1 hour
        "language_detection": 1800,      # 30 minutes
        "response_generation": 600       # 10 minutes
    }
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Model training configuration
    training_config: Dict[str, Any] = {
        "intent_classifier": {
            "algorithm": "svm",
            "test_size": 0.2,
            "random_state": 42,
            "cross_validation_folds": 5
        },
        "entity_extractor": {
            "algorithm": "crf",
            "features": ["word", "pos", "shape", "context"],
            "context_window": 2
        }
    }
    
    # API rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    
    # Monitoring and metrics
    enable_metrics: bool = True
    metrics_port: int = 8002
    
    # Feature flags
    features: Dict[str, bool] = {
        "multilingual_support": True,
        "entity_extraction": True,
        "response_caching": True,
        "model_retraining": True,
        "analytics_logging": True
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Intent mapping for different languages
INTENT_MAPPINGS = {
    "en": {
        "transport_query": [
            "bus", "route", "schedule", "fare", "transport", "travel",
            "departure", "arrival", "ticket", "journey", "trip"
        ],
        "entertainment_request": [
            "joke", "fun", "entertainment", "quiz", "game", "story",
            "music", "news", "fact", "trivia"
        ],
        "support_request": [
            "help", "support", "problem", "issue", "question", "assistance",
            "contact", "report", "bug", "error"
        ],
        "greeting": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "greetings"
        ],
        "goodbye": [
            "bye", "goodbye", "see you", "farewell", "take care",
            "until next time"
        ]
    },
    "rw": {
        "transport_query": [
            "bisi", "inzira", "gahunda", "ikiguzi", "ingendo",
            "kugenda", "kuza", "gutaha"
        ],
        "entertainment_request": [
            "urwenya", "kwinezeza", "ikibazo", "umukino", "inkuru",
            "amakuru", "ukuri"
        ],
        "support_request": [
            "gufasha", "ikibazo", "ubufasha", "kuvugana",
            "raporo", "ikosa"
        ],
        "greeting": [
            "muraho", "mwaramutse", "mwiriwe", "bite", "amahoro"
        ],
        "goodbye": [
            "murabeho", "tugire amahoro", "tuzabonana"
        ]
    },
    "fr": {
        "transport_query": [
            "bus", "autobus", "route", "horaire", "tarif", "transport",
            "voyage", "départ", "arrivée", "billet"
        ],
        "entertainment_request": [
            "blague", "amusement", "divertissement", "quiz", "jeu",
            "histoire", "musique", "nouvelles", "fait"
        ],
        "support_request": [
            "aide", "support", "problème", "question", "assistance",
            "contact", "rapport", "erreur"
        ],
        "greeting": [
            "bonjour", "salut", "bonsoir", "bonne journée"
        ],
        "goodbye": [
            "au revoir", "à bientôt", "bonne journée", "salut"
        ]
    }
}

# Response templates for different intents and languages
RESPONSE_TEMPLATES = {
    "en": {
        "transport_query": [
            "I can help you with transport information. What would you like to know?",
            "Let me find transport details for you.",
            "I'm here to help with your travel needs."
        ],
        "entertainment_request": [
            "I'd be happy to entertain you! What would you like?",
            "Let's have some fun! What can I do for you?",
            "Entertainment time! How can I brighten your day?"
        ],
        "support_request": [
            "I'm here to help. What can I assist you with?",
            "Let me help you with your question.",
            "How can I support you today?"
        ],
        "greeting": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Greetings! I'm here to assist you."
        ],
        "goodbye": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Farewell! Come back anytime."
        ],
        "general": [
            "I understand. Let me help you with that.",
            "That's interesting. How can I assist?",
            "I'm here to help. Could you be more specific?"
        ]
    },
    "rw": {
        "transport_query": [
            "Nshobora kugufasha kubijyanye n'ingendo. Ni iki ushaka kumenya?",
            "Reka nkushoze amakuru y'ingendo.",
            "Ndi hano gufasha mu bibazo by'ingendo."
        ],
        "entertainment_request": [
            "Nishimiye kugukeza! Ni iki ushaka?",
            "Reka twinezeze! Ni iki nkora?",
            "Igihe cy'kwishimisha! Ni gute nkoresha umunsi wawe?"
        ],
        "support_request": [
            "Ndi hano gufasha. Ni iki nkora?",
            "Reka nkugufashe n'ikibazo cyawe.",
            "Ni gute nkugufasha uyu munsi?"
        ],
        "greeting": [
            "Muraho! Ni gute nkugufasha uyu munsi?",
            "Mwaramutse! Ni iki nkora?",
            "Amahoro! Ndi hano kugufasha."
        ],
        "goodbye": [
            "Murabeho! Mwumve umunsi mwiza!",
            "Tuzabonana! Mwicuze!",
            "Tugire amahoro! Mugaruke ubwoba ubwo."
        ],
        "general": [
            "Ndabyumva. Reka nkugufashe.",
            "Ni byiza. Ni gute nkugufasha?",
            "Ndi hano gufasha. Waba ushobora kugereranya?"
        ]
    },
    "fr": {
        "transport_query": [
            "Je peux vous aider avec les informations de transport. Que voulez-vous savoir?",
            "Laissez-moi trouver les détails de transport pour vous.",
            "Je suis là pour vous aider avec vos besoins de voyage."
        ],
        "entertainment_request": [
            "Je serais ravi de vous divertir! Que voulez-vous?",
            "Amusons-nous! Que puis-je faire pour vous?",
            "C'est l'heure du divertissement! Comment puis-je égayer votre journée?"
        ],
        "support_request": [
            "Je suis là pour aider. Avec quoi puis-je vous assister?",
            "Laissez-moi vous aider avec votre question.",
            "Comment puis-je vous soutenir aujourd'hui?"
        ],
        "greeting": [
            "Bonjour! Comment puis-je vous aider aujourd'hui?",
            "Salut! Que puis-je faire pour vous?",
            "Salutations! Je suis là pour vous assister."
        ],
        "goodbye": [
            "Au revoir! Passez une excellente journée!",
            "À plus tard! Prenez soin de vous!",
            "Adieu! Revenez quand vous voulez."
        ],
        "general": [
            "Je comprends. Laissez-moi vous aider avec ça.",
            "C'est intéressant. Comment puis-je vous assister?",
            "Je suis là pour aider. Pourriez-vous être plus précis?"
        ]
    }
}