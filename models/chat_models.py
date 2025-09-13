"""
Pydantic models for chat API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class User(BaseModel):
    """User information model"""
    id: Optional[str] = None
    preferredLanguage: Optional[str] = "en"
    name: Optional[str] = None
    email: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat processing"""
    message: str = Field(..., min_length=1, max_length=1000, description="The user's message")
    language: Optional[str] = Field(None, description="Language code (en, rw, fr)")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    user: Optional[User] = Field(None, description="User information")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Request timestamp")


class ChatResponse(BaseModel):
    """Response model for chat processing"""
    response: str = Field(..., description="The AI's response")
    intent: str = Field(..., description="Detected intent")
    entities: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Extracted entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    language: str = Field(..., description="Detected/used language")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="Timestamp of health check")
    services: Optional[Dict[str, str]] = Field(None, description="Individual service statuses")


class LanguageDetectionRequest(BaseModel):
    """Request model for language detection"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to detect language for")


class LanguageDetectionResponse(BaseModel):
    """Response model for language detection"""
    text: str = Field(..., description="Original text")
    language: str = Field(..., description="Detected language code")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection confidence")
    timestamp: str = Field(..., description="Detection timestamp")


class TranslationRequest(BaseModel):
    """Request model for translation"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to translate")
    source_lang: str = Field(..., description="Source language code")
    target_lang: str = Field(..., description="Target language code")


class TranslationResponse(BaseModel):
    """Response model for translation"""
    original_text: str = Field(..., description="Original text")
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Translation confidence")
    timestamp: str = Field(..., description="Translation timestamp")


class EntityExtractionRequest(BaseModel):
    """Request model for entity extraction"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to extract entities from")
    intent: Optional[str] = Field(None, description="Detected intent for context")


class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction"""
    text: str = Field(..., description="Original text")
    intent: Optional[str] = Field(None, description="Provided intent")
    entities: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Extracted entities")
    timestamp: str = Field(..., description="Extraction timestamp")


class IntentClassificationRequest(BaseModel):
    """Request model for intent classification"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to classify")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class IntentClassificationResponse(BaseModel):
    """Response model for intent classification"""
    text: str = Field(..., description="Original text")
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative intents")
    timestamp: str = Field(..., description="Classification timestamp")


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    nlp_models: Dict[str, Any] = Field(..., description="NLP model information")
    intent_model: Dict[str, Any] = Field(..., description="Intent classifier information")
    entity_model: Dict[str, Any] = Field(..., description="Entity extractor information")
    timestamp: str = Field(..., description="Information timestamp")


class RetrainRequest(BaseModel):
    """Request model for model retraining"""
    model_type: str = Field(..., description="Type of model to retrain (intent, entity)")
    force: bool = Field(False, description="Force retraining even if model is recent")


class RetrainResponse(BaseModel):
    """Response model for model retraining"""
    message: str = Field(..., description="Retraining status message")
    model_type: str = Field(..., description="Type of model being retrained")
    timestamp: str = Field(..., description="Retraining timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
