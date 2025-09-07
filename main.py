"""
Translinka AI/ML Service
FastAPI-based machine learning service for NLP processing, intent recognition, and multilingual support.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
from typing import Optional, Dict, Any, List
import asyncio
import logging
from datetime import datetime

from config.settings import Settings
from services.nlp_service import NLPService
from services.intent_classifier import IntentClassifier
from services.entity_extractor import EntityExtractor
from services.response_generator import ResponseGenerator
from services.translation_service import TranslationService
from models.chat_models import ChatRequest, ChatResponse, HealthResponse
from utils.logger import setup_logger
from utils.cache import CacheManager

# Global variables for services
nlp_service: Optional[NLPService] = None
intent_classifier: Optional[IntentClassifier] = None
entity_extractor: Optional[EntityExtractor] = None
response_generator: Optional[ResponseGenerator] = None
translation_service: Optional[TranslationService] = None
cache_manager: Optional[CacheManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup ML services"""
    global nlp_service, intent_classifier, entity_extractor, response_generator, translation_service, cache_manager
    
    logger = setup_logger()
    settings = Settings()
    
    try:
        logger.info("🚀 Starting Translinka ML Service...")
        
        # Initialize cache manager
        cache_manager = CacheManager(settings.redis_url)
        await cache_manager.connect()
        logger.info("✅ Cache manager initialized")
        
        # Initialize NLP service
        nlp_service = NLPService(settings)
        await nlp_service.initialize()
        logger.info("✅ NLP service initialized")
        
        # Initialize intent classifier
        intent_classifier = IntentClassifier(settings, cache_manager)
        await intent_classifier.initialize()
        logger.info("✅ Intent classifier initialized")
        
        # Initialize entity extractor
        entity_extractor = EntityExtractor(settings, nlp_service)
        await entity_extractor.initialize()
        logger.info("✅ Entity extractor initialized")
        
        # Initialize response generator
        response_generator = ResponseGenerator(settings, cache_manager)
        await response_generator.initialize()
        logger.info("✅ Response generator initialized")
        
        # Initialize translation service
        translation_service = TranslationService(settings, cache_manager)
        await translation_service.initialize()
        logger.info("✅ Translation service initialized")
        
        logger.info("🎉 All ML services initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up ML services...")
        if cache_manager:
            await cache_manager.close()
        logger.info("✅ ML services cleanup completed")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Translinka AI/ML Service",
    description="Machine Learning and NLP service for the Translinka AI Chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logger = setup_logger()

# Dependency to get services
def get_services():
    """Dependency to get all ML services"""
    if not all([nlp_service, intent_classifier, entity_extractor, response_generator, translation_service]):
        raise HTTPException(status_code=503, detail="ML services not initialized")
    
    return {
        'nlp': nlp_service,
        'intent': intent_classifier,
        'entity': entity_extractor,
        'response': response_generator,
        'translation': translation_service,
        'cache': cache_manager
    }

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Translinka AI/ML Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if all services are initialized
        if not all([nlp_service, intent_classifier, entity_extractor, response_generator, translation_service]):
            return HealthResponse(
                status="unhealthy",
                message="Some services not initialized",
                timestamp=datetime.now().isoformat()
            )
        
        # Check cache connection
        if cache_manager and not await cache_manager.is_connected():
            return HealthResponse(
                status="degraded",
                message="Cache service unavailable",
                timestamp=datetime.now().isoformat()
            )
        
        return HealthResponse(
            status="healthy",
            message="All services operational",
            timestamp=datetime.now().isoformat(),
            services={
                "nlp": "healthy",
                "intent_classifier": "healthy",
                "entity_extractor": "healthy",
                "response_generator": "healthy",
                "translation": "healthy",
                "cache": "healthy" if cache_manager and await cache_manager.is_connected() else "unavailable"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.post("/chat/process", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    services: Dict = Depends(get_services)
):
    """Process a chat message and return AI response"""
    try:
        start_time = datetime.now()
        logger.info(f"Processing chat message: {request.message[:50]}...")
        
        # Detect language if not provided
        detected_language = request.language
        if not detected_language:
            detected_language = await services['nlp'].detect_language(request.message)
            logger.info(f"Detected language: {detected_language}")
        
        # Translate to English for processing if needed
        original_message = request.message
        processing_message = original_message
        
        if detected_language != 'en':
            processing_message = await services['translation'].translate(
                original_message, 
                source_lang=detected_language, 
                target_lang='en'
            )
            logger.info(f"Translated for processing: {processing_message}")
        
        # Extract intent
        intent_result = await services['intent'].classify_intent(
            processing_message, 
            context=request.context
        )
        logger.info(f"Intent classified: {intent_result['intent']} (confidence: {intent_result['confidence']})")
        
        # Extract entities
        entities = await services['entity'].extract_entities(
            processing_message,
            intent=intent_result['intent']
        )
        logger.info(f"Entities extracted: {entities}")
        
        # Generate response
        response_data = await services['response'].generate_response(
            intent=intent_result['intent'],
            entities=entities,
            original_message=original_message,
            user_context=request.context,
            language=detected_language,
            confidence=intent_result['confidence']
        )
        
        # Translate response back if needed
        final_response = response_data['response']
        if detected_language != 'en' and response_data.get('needs_translation', True):
            final_response = await services['translation'].translate(
                response_data['response'],
                source_lang='en',
                target_lang=detected_language
            )
            logger.info(f"Response translated to {detected_language}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log analytics in background
        background_tasks.add_task(
            log_chat_analytics,
            request=request,
            intent=intent_result['intent'],
            confidence=intent_result['confidence'],
            processing_time=processing_time,
            language=detected_language
        )
        
        return ChatResponse(
            response=final_response,
            intent=intent_result['intent'],
            entities=entities,
            confidence=intent_result['confidence'],
            language=detected_language,
            processing_time_ms=int(processing_time),
            metadata={
                **response_data.get('metadata', {}),
                'original_language': detected_language,
                'translated_for_processing': detected_language != 'en',
                'response_translated': detected_language != 'en' and response_data.get('needs_translation', True)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )

@app.post("/nlp/detect-language")
async def detect_language(
    text: str,
    services: Dict = Depends(get_services)
):
    """Detect the language of given text"""
    try:
        language = await services['nlp'].detect_language(text)
        return {
            "text": text,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/translate")
async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    services: Dict = Depends(get_services)
):
    """Translate text between languages"""
    try:
        translated = await services['translation'].translate(text, source_lang, target_lang)
        return {
            "original_text": text,
            "translated_text": translated,
            "source_language": source_lang,
            "target_language": target_lang,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/extract-entities")
async def extract_entities(
    text: str,
    intent: Optional[str] = None,
    services: Dict = Depends(get_services)
):
    """Extract entities from text"""
    try:
        entities = await services['entity'].extract_entities(text, intent)
        return {
            "text": text,
            "intent": intent,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/classify-intent")
async def classify_intent(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    services: Dict = Depends(get_services)
):
    """Classify intent of text"""
    try:
        result = await services['intent'].classify_intent(text, context)
        return {
            "text": text,
            "intent": result['intent'],
            "confidence": result['confidence'],
            "alternatives": result.get('alternatives', []),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_model_info(services: Dict = Depends(get_services)):
    """Get information about loaded models"""
    try:
        return {
            "nlp_models": await services['nlp'].get_model_info(),
            "intent_model": await services['intent'].get_model_info(),
            "entity_model": await services['entity'].get_model_info(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/retrain")
async def retrain_models(
    background_tasks: BackgroundTasks,
    model_type: str = "intent",
    services: Dict = Depends(get_services)
):
    """Trigger model retraining (background task)"""
    try:
        if model_type == "intent":
            background_tasks.add_task(services['intent'].retrain_model)
        elif model_type == "entity":
            background_tasks.add_task(services['entity'].retrain_model)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        return {
            "message": f"Retraining {model_type} model initiated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def log_chat_analytics(
    request: ChatRequest,
    intent: str,
    confidence: float,
    processing_time: float,
    language: str
):
    """Log chat analytics for monitoring and improvement"""
    try:
        analytics_data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user.get("id") if request.user else None,
            "message_length": len(request.message),
            "intent": intent,
            "confidence": confidence,
            "language": language,
            "processing_time_ms": processing_time,
            "context_provided": bool(request.context)
        }
        
        # In production, send this to analytics service or database
        logger.info(f"Chat analytics: {analytics_data}")
        
    except Exception as e:
        logger.error(f"Failed to log analytics: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )