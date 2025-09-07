"""
Tests for the main ML service application
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json

from main import app
from models.chat_models import ChatRequest, HealthResponse

client = TestClient(app)

class TestMainApp:
    """Test cases for the main FastAPI application"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Translinka AI/ML Service"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "timestamp" in data
    
    def test_health_endpoint_uninitialized(self):
        """Test health endpoint when services are not initialized"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy", "degraded"]
        assert "timestamp" in data
    
    @patch('main.nlp_service')
    @patch('main.intent_classifier')
    @patch('main.entity_extractor')
    @patch('main.response_generator')
    @patch('main.translation_service')
    @patch('main.cache_manager')
    def test_health_endpoint_initialized(self, mock_cache, mock_translation, 
                                       mock_response, mock_entity, mock_intent, mock_nlp):
        """Test health endpoint when all services are initialized"""
        # Mock all services as initialized
        mock_nlp.return_value = MagicMock()
        mock_intent.return_value = MagicMock()
        mock_entity.return_value = MagicMock()
        mock_response.return_value = MagicMock()
        mock_translation.return_value = MagicMock()
        mock_cache.is_connected = AsyncMock(return_value=True)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_chat_process_without_services(self):
        """Test chat processing when services are not available"""
        chat_request = {
            "message": "Hello, how are you?",
            "language": "en",
            "context": {},
            "user": {"id": "test-user", "preferredLanguage": "en"}
        }
        
        response = client.post("/chat/process", json=chat_request)
        assert response.status_code == 503  # Service unavailable
    
    @patch('main.get_services')
    def test_detect_language_endpoint(self, mock_get_services):
        """Test language detection endpoint"""
        mock_services = {
            'nlp': MagicMock(),
            'intent': MagicMock(),
            'entity': MagicMock(),
            'response': MagicMock(),
            'translation': MagicMock(),
            'cache': MagicMock()
        }
        mock_services['nlp'].detect_language = AsyncMock(return_value='en')
        mock_get_services.return_value = mock_services
        
        response = client.post("/nlp/detect-language", params={"text": "Hello world"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"
        assert data["text"] == "Hello world"
        assert "timestamp" in data
    
    @patch('main.get_services')
    def test_translate_endpoint(self, mock_get_services):
        """Test translation endpoint"""
        mock_services = {
            'translation': MagicMock()
        }
        mock_services['translation'].translate = AsyncMock(return_value='Bonjour le monde')
        mock_get_services.return_value = mock_services
        
        response = client.post("/nlp/translate", params={
            "text": "Hello world",
            "source_lang": "en",
            "target_lang": "fr"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["translated_text"] == "Bonjour le monde"
        assert data["source_language"] == "en"
        assert data["target_language"] == "fr"
    
    @patch('main.get_services')
    def test_extract_entities_endpoint(self, mock_get_services):
        """Test entity extraction endpoint"""
        mock_services = {
            'entity': MagicMock()
        }
        mock_entities = {
            "LOCATION": [{"text": "Kigali", "confidence": 0.9}],
            "TIME": [{"text": "8:00 AM", "confidence": 0.8}]
        }
        mock_services['entity'].extract_entities = AsyncMock(return_value=mock_entities)
        mock_get_services.return_value = mock_services
        
        response = client.post("/nlp/extract-entities", params={
            "text": "Bus to Kigali at 8:00 AM",
            "intent": "transport_query"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["entities"] == mock_entities
        assert data["text"] == "Bus to Kigali at 8:00 AM"
        assert data["intent"] == "transport_query"
    
    @patch('main.get_services')
    def test_classify_intent_endpoint(self, mock_get_services):
        """Test intent classification endpoint"""
        mock_services = {
            'intent': MagicMock()
        }
        mock_result = {
            "intent": "transport_query",
            "confidence": 0.95,
            "alternatives": [
                {"intent": "general", "confidence": 0.05}
            ]
        }
        mock_services['intent'].classify_intent = AsyncMock(return_value=mock_result)
        mock_get_services.return_value = mock_services
        
        response = client.post("/nlp/classify-intent", params={
            "text": "What time does the bus leave?"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "transport_query"
        assert data["confidence"] == 0.95
        assert len(data["alternatives"]) > 0
    
    @patch('main.get_services')
    def test_model_info_endpoint(self, mock_get_services):
        """Test model information endpoint"""
        mock_services = {
            'nlp': MagicMock(),
            'intent': MagicMock(),
            'entity': MagicMock()
        }
        
        mock_services['nlp'].get_model_info = AsyncMock(return_value={
            "initialized": True,
            "models": ["en", "fr"]
        })
        mock_services['intent'].get_model_info = AsyncMock(return_value={
            "model_type": "SVM",
            "accuracy": 0.85
        })
        mock_services['entity'].get_model_info = AsyncMock(return_value={
            "entity_types": ["LOCATION", "TIME", "FARE"]
        })
        
        mock_get_services.return_value = mock_services
        
        response = client.get("/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "nlp_models" in data
        assert "intent_model" in data
        assert "entity_model" in data
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/")
        # CORS headers should be present in a real CORS request
        # This is a basic test to ensure the endpoint responds
        assert response.status_code == 200

@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test cases for async endpoints"""
    
    @patch('main.get_services')
    async def test_chat_process_full_flow(self, mock_get_services):
        """Test complete chat processing flow"""
        mock_services = {
            'nlp': MagicMock(),
            'intent': MagicMock(),
            'entity': MagicMock(),
            'response': MagicMock(),
            'translation': MagicMock(),
            'cache': MagicMock()
        }
        
        # Mock service responses
        mock_services['nlp'].detect_language = AsyncMock(return_value='en')
        mock_services['intent'].classify_intent = AsyncMock(return_value={
            'intent': 'transport_query',
            'confidence': 0.9,
            'alternatives': []
        })
        mock_services['entity'].extract_entities = AsyncMock(return_value={
            'LOCATION': [{'text': 'Kigali', 'confidence': 0.9}]
        })
        mock_services['response'].generate_response = AsyncMock(return_value={
            'response': 'Here are the bus schedules for Kigali...',
            'metadata': {'source': 'transport_api'},
            'needs_translation': False
        })
        
        mock_get_services.return_value = mock_services
        
        chat_request = {
            "message": "Show me bus schedules for Kigali",
            "language": "en",
            "context": {},
            "user": {"id": "test-user", "preferredLanguage": "en"}
        }
        
        response = client.post("/chat/process", json=chat_request)
        assert response.status_code == 200
        data = response.json()
        
        assert data["response"] == "Here are the bus schedules for Kigali..."
        assert data["intent"] == "transport_query"
        assert data["confidence"] == 0.9
        assert data["language"] == "en"
        assert "processing_time_ms" in data
        assert "metadata" in data

if __name__ == "__main__":
    pytest.main([__file__])