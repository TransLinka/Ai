"""
Entity extraction service for identifying and extracting entities from text
"""

import re
import json
import os
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from datetime import datetime, timedelta
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

from config.settings import Settings
from utils.logger import ml_logger, log_ml_operation
from services.nlp_service import NLPService

class EntityExtractor:
    """
    Service for extracting entities from text using rule-based and ML approaches
    """
    
    def __init__(self, settings: Settings, nlp_service: NLPService):
        """
        Initialize entity extractor
        
        Args:
            settings: Application settings
            nlp_service: NLP service instance
        """
        self.settings = settings
        self.nlp_service = nlp_service
        self.matchers = {}
        self.patterns = {}
        self.custom_entities = {}
        self._initialized = False
        
        # Entity extraction patterns
        self.entity_patterns = {
            'TIME': [
                r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b',
                r'\b(\d{1,2})\s*(AM|PM|am|pm)\b',
                r'\b(morning|afternoon|evening|night)\b',
                r'\b(early|late)\s+(morning|afternoon|evening)\b'
            ],
            'DATE': [
                r'\b(today|tomorrow|yesterday)\b',
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b',
                r'\b(\d{1,2})-(\d{1,2})-(\d{2,4})\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b'
            ],
            'LOCATION': [
                r'\b(kigali|huye|musanze|rubavu|nyagatare|butare|gisenyi|ruhengeri)\b',
                r'\b(from|to|via|through)\s+([A-Za-z\s]+)\b',
                r'\b(district|sector|cell)\s+([A-Za-z\s]+)\b'
            ],
            'TRANSPORT_TYPE': [
                r'\b(bus|taxi|moto|bicycle|car|matatu|coaster)\b',
                r'\b(public\s+transport|private\s+transport)\b'
            ],
            'FARE': [
                r'\b(\d+)\s*(rwf|francs?|frw)\b',
                r'\b(rwf|frw)\s*(\d+)\b',
                r'\b(\d+)\s*(shillings?)\b'
            ],
            'ROUTE': [
                r'\broute\s+([A-Za-z0-9\-]+)\b',
                r'\bline\s+([A-Za-z0-9\-]+)\b',
                r'\b([A-Za-z]+)\s*-\s*([A-Za-z]+)\s+route\b'
            ]
        }
    
    async def initialize(self):
        """Initialize the entity extractor"""
        try:
            start_time = datetime.now()
            ml_logger.info("Initializing entity extractor...")
            
            # Load custom entities if available
            await self._load_custom_entities()
            
            # Initialize spaCy matchers for different languages
            await self._initialize_matchers()
            
            # Compile regex patterns
            self._compile_patterns()
            
            self._initialized = True
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            log_ml_operation("entity_extractor_init", duration, True, {
                "entity_types": len(self.entity_patterns),
                "custom_entities": len(self.custom_entities)
            })
            
            ml_logger.info("Entity extractor initialized successfully")
            
        except Exception as e:
            ml_logger.error(f"Failed to initialize entity extractor: {e}")
            log_ml_operation("entity_extractor_init", 0, False, {"error": str(e)})
            raise
    
    async def _load_custom_entities(self):
        """Load custom entities from configuration file"""
        try:
            if os.path.exists(self.settings.custom_entities_path):
                with open(self.settings.custom_entities_path, 'r', encoding='utf-8') as f:
                    self.custom_entities = json.load(f)
                ml_logger.info(f"Loaded {len(self.custom_entities)} custom entity types")
            else:
                # Create default custom entities
                self.custom_entities = {
                    "RWANDA_LOCATIONS": [
                        "kigali", "huye", "musanze", "rubavu", "nyagatare", "butare",
                        "gisenyi", "ruhengeri", "muhanga", "karongi", "rusizi", "ngoma",
                        "kayonza", "kirehe", "nyanza", "gicumbi", "rulindo", "gatsibo",
                        "nyaruguru", "nyamagabe", "burera", "gakenke", "rwamagana",
                        "bugesera", "ruhango", "nyarugenge", "gasabo", "kicukiro"
                    ],
                    "TRANSPORT_OPERATORS": [
                        "kbs", "rwanda federation of transport cooperatives",
                        "rftc", "horizon", "volcano", "trinity", "bisate", "sotra"
                    ],
                    "ENTERTAINMENT_TYPES": [
                        "joke", "story", "fact", "quiz", "news", "music", "game",
                        "trivia", "riddle", "motivation", "inspiration"
                    ]
                }
                await self._save_custom_entities()
        except Exception as e:
            ml_logger.warning(f"Failed to load custom entities: {e}")
            self.custom_entities = {}
    
    async def _save_custom_entities(self):
        """Save custom entities to file"""
        try:
            os.makedirs(os.path.dirname(self.settings.custom_entities_path), exist_ok=True)
            with open(self.settings.custom_entities_path, 'w', encoding='utf-8') as f:
                json.dump(self.custom_entities, f, indent=2, ensure_ascii=False)
        except Exception as e:
            ml_logger.warning(f"Failed to save custom entities: {e}")
    
    async def _initialize_matchers(self):
        """Initialize spaCy matchers for different languages"""
        if not self.nlp_service.is_initialized():
            return
        
        for lang in ['en', 'fr']:
            if f'spacy_{lang}' in self.nlp_service.models:
                nlp = self.nlp_service.models[f'spacy_{lang}']
                matcher = Matcher(nlp.vocab)
                
                # Add location patterns
                location_patterns = [
                    [{"LOWER": {"IN": self.custom_entities.get("RWANDA_LOCATIONS", [])}}],
                    [{"ENT_TYPE": "GPE"}],
                    [{"ENT_TYPE": "LOC"}]
                ]
                matcher.add("LOCATION", location_patterns)
                
                # Add transport patterns
                transport_patterns = [
                    [{"LOWER": {"IN": ["bus", "taxi", "moto", "car", "matatu"]}}],
                    [{"LOWER": "public"}, {"LOWER": "transport"}]
                ]
                matcher.add("TRANSPORT_TYPE", transport_patterns)
                
                # Add time patterns
                time_patterns = [
                    [{"LIKE_NUM": True}, {"TEXT": ":"}, {"LIKE_NUM": True}],
                    [{"LOWER": {"IN": ["morning", "afternoon", "evening", "night"]}}]
                ]
                matcher.add("TIME", time_patterns)
                
                self.matchers[lang] = matcher
    
    def _compile_patterns(self):
        """Compile regex patterns for entity extraction"""
        self.compiled_patterns = {}
        for entity_type, patterns in self.entity_patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    async def extract_entities(
        self,
        text: str,
        intent: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from text
        
        Args:
            text: Input text
            intent: Known intent for context
            language: Text language
            
        Returns:
            Dictionary of extracted entities
        """
        if not self._initialized:
            raise ValueError("Entity extractor not initialized")
        
        if not text or not text.strip():
            return {}
        
        try:
            start_time = datetime.now()
            
            if not language:
                language = await self.nlp_service.detect_language(text)
            
            entities = {}
            
            # Extract using spaCy if available
            if language in self.matchers:
                spacy_entities = await self._extract_with_spacy(text, language)
                entities.update(spacy_entities)
            
            # Extract using regex patterns
            regex_entities = await self._extract_with_regex(text)
            entities.update(regex_entities)
            
            # Extract custom entities
            custom_entities = await self._extract_custom_entities(text)
            entities.update(custom_entities)
            
            # Apply intent-specific processing
            if intent:
                entities = self._apply_intent_context(entities, intent, text)
            
            # Post-process and validate entities
            entities = self._post_process_entities(entities, text)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log_ml_operation("entity_extraction", duration, True, {
                "entities_found": len(entities),
                "text_length": len(text),
                "language": language,
                "intent": intent
            })
            
            return entities
            
        except Exception as e:
            ml_logger.error(f"Entity extraction error: {e}")
            log_ml_operation("entity_extraction", 0, False, {"error": str(e)})
            return {}
    
    async def _extract_with_spacy(self, text: str, language: str) -> Dict[str, Any]:
        """Extract entities using spaCy NER and matcher"""
        entities = {}
        
        try:
            nlp = self.nlp_service.models[f'spacy_{language}']
            matcher = self.matchers[language]
            
            # Process text
            doc = await asyncio.get_event_loop().run_in_executor(None, nlp, text)
            
            # Extract named entities
            for ent in doc.ents:
                entity_type = self._map_spacy_entity_type(ent.label_)
                if entity_type:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    
                    entities[entity_type].append({
                        "text": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.8,  # Default confidence for spaCy entities
                        "source": "spacy_ner"
                    })
            
            # Extract using patterns
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                label = nlp.vocab.strings[match_id]
                
                if label not in entities:
                    entities[label] = []
                
                entities[label].append({
                    "text": span.text,
                    "start": span.start_char,
                    "end": span.end_char,
                    "confidence": 0.9,  # Higher confidence for pattern matches
                    "source": "spacy_matcher"
                })
            
        except Exception as e:
            ml_logger.warning(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    async def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract entities using regex patterns"""
        entities = {}
        
        for entity_type, patterns in self.compiled_patterns.items():
            matches = []
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append({
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.7,  # Moderate confidence for regex
                        "source": "regex",
                        "groups": match.groups()
                    })
            
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    async def _extract_custom_entities(self, text: str) -> Dict[str, Any]:
        """Extract custom entities from text"""
        entities = {}
        text_lower = text.lower()
        
        for entity_type, entity_list in self.custom_entities.items():
            matches = []
            
            for entity_value in entity_list:
                entity_lower = entity_value.lower()
                start_pos = 0
                
                while True:
                    pos = text_lower.find(entity_lower, start_pos)
                    if pos == -1:
                        break
                    
                    # Check word boundaries
                    if (pos == 0 or not text[pos-1].isalnum()) and \
                       (pos + len(entity_value) >= len(text) or not text[pos + len(entity_value)].isalnum()):
                        
                        matches.append({
                            "text": text[pos:pos + len(entity_value)],
                            "start": pos,
                            "end": pos + len(entity_value),
                            "confidence": 0.9,
                            "source": "custom_entities",
                            "entity_type": entity_type
                        })
                    
                    start_pos = pos + 1
            
            if matches:
                # Map custom entity types to standard types
                standard_type = self._map_custom_entity_type(entity_type)
                entities[standard_type] = matches
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            'GPE': 'LOCATION',        # Geopolitical entity
            'LOC': 'LOCATION',        # Location
            'TIME': 'TIME',           # Time
            'DATE': 'DATE',           # Date
            'MONEY': 'FARE',          # Money
            'PERSON': 'PERSON',       # Person
            'ORG': 'ORGANIZATION',    # Organization
            'PRODUCT': 'TRANSPORT_TYPE'  # Product (for transport types)
        }
        return mapping.get(spacy_label)
    
    def _map_custom_entity_type(self, custom_type: str) -> str:
        """Map custom entity types to standard types"""
        mapping = {
            'RWANDA_LOCATIONS': 'LOCATION',
            'TRANSPORT_OPERATORS': 'ORGANIZATION',
            'ENTERTAINMENT_TYPES': 'ENTERTAINMENT_TYPE'
        }
        return mapping.get(custom_type, custom_type)
    
    def _apply_intent_context(
        self,
        entities: Dict[str, Any],
        intent: str,
        text: str
    ) -> Dict[str, Any]:
        """Apply intent-specific context to entity extraction"""
        
        # Transport-specific processing
        if intent == "transport_query":
            # Look for origin/destination patterns
            origin_dest = self._extract_origin_destination(text)
            if origin_dest:
                entities.update(origin_dest)
            
            # Enhance location entities with transport context
            if 'LOCATION' in entities:
                for location in entities['LOCATION']:
                    # Determine if it's origin or destination based on context
                    context = self._determine_location_context(location['text'], text)
                    location['context'] = context
        
        # Entertainment-specific processing
        elif intent == "entertainment_request":
            # Look for specific entertainment requests
            entertainment_type = self._extract_entertainment_type(text)
            if entertainment_type:
                entities['ENTERTAINMENT_TYPE'] = [entertainment_type]
        
        return entities
    
    def _extract_origin_destination(self, text: str) -> Dict[str, Any]:
        """Extract origin and destination from transport queries"""
        entities = {}
        
        # Patterns for origin-destination
        patterns = [
            r'\b(?:from|leaving)\s+([A-Za-z\s]+)\s+(?:to|going\s+to|heading\s+to)\s+([A-Za-z\s]+)\b',
            r'\b([A-Za-z\s]+)\s+(?:to|->|-)\s+([A-Za-z\s]+)\b',
            r'\b(?:bus|taxi|transport)\s+(?:from\s+)?([A-Za-z\s]+)\s+(?:to\s+)?([A-Za-z\s]+)\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                origin = match.group(1).strip()
                destination = match.group(2).strip()
                
                if self._is_valid_location(origin):
                    entities['ORIGIN'] = [{
                        "text": origin,
                        "confidence": 0.8,
                        "source": "context_extraction"
                    }]
                
                if self._is_valid_location(destination):
                    entities['DESTINATION'] = [{
                        "text": destination,
                        "confidence": 0.8,
                        "source": "context_extraction"
                    }]
                
                break
        
        return entities
    
    def _determine_location_context(self, location: str, text: str) -> str:
        """Determine if a location is origin or destination"""
        text_lower = text.lower()
        location_lower = location.lower()
        
        # Find position of location in text
        pos = text_lower.find(location_lower)
        if pos == -1:
            return "unknown"
        
        # Check preceding words
        preceding = text_lower[:pos].split()[-3:]  # Last 3 words before location
        following = text_lower[pos + len(location):].split()[:3]  # First 3 words after location
        
        origin_indicators = ['from', 'leaving', 'departing', 'starting']
        destination_indicators = ['to', 'going', 'heading', 'arriving', 'reaching']
        
        for word in preceding + following:
            if word in origin_indicators:
                return "origin"
            elif word in destination_indicators:
                return "destination"
        
        return "unknown"
    
    def _extract_entertainment_type(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract specific entertainment type from request"""
        text_lower = text.lower()
        
        entertainment_patterns = {
            'joke': ['joke', 'funny', 'humor', 'laugh'],
            'fact': ['fact', 'information', 'knowledge', 'learn'],
            'quiz': ['quiz', 'question', 'test', 'challenge'],
            'news': ['news', 'updates', 'latest', 'current'],
            'music': ['music', 'song', 'audio'],
            'story': ['story', 'tale', 'narrative']
        }
        
        for ent_type, keywords in entertainment_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        "text": ent_type,
                        "confidence": 0.8,
                        "source": "entertainment_context"
                    }
        
        return None
    
    def _is_valid_location(self, location: str) -> bool:
        """Check if a string is a valid location"""
        location_lower = location.lower().strip()
        
        # Check against known locations
        known_locations = self.custom_entities.get("RWANDA_LOCATIONS", [])
        if location_lower in [loc.lower() for loc in known_locations]:
            return True
        
        # Basic validation rules
        if len(location_lower) < 2 or len(location_lower) > 50:
            return False
        
        # Should contain mostly letters
        alpha_ratio = sum(c.isalpha() for c in location_lower) / len(location_lower)
        if alpha_ratio < 0.7:
            return False
        
        return True
    
    def _post_process_entities(
        self,
        entities: Dict[str, Any],
        text: str
    ) -> Dict[str, Any]:
        """Post-process and clean extracted entities"""
        processed = {}
        
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
            
            # Remove duplicates and overlapping entities
            cleaned_entities = self._remove_overlapping_entities(entity_list)
            
            # Validate and clean entity values
            valid_entities = []
            for entity in cleaned_entities:
                if self._validate_entity(entity, entity_type):
                    # Clean entity text
                    entity['text'] = entity['text'].strip()
                    valid_entities.append(entity)
            
            if valid_entities:
                processed[entity_type] = valid_entities
        
        return processed
    
    def _remove_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities, keeping the one with higher confidence"""
        if len(entities) <= 1:
            return entities
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))
        
        result = []
        for entity in sorted_entities:
            overlapping = False
            
            for existing in result:
                # Check for overlap
                if (entity.get('start', 0) < existing.get('end', 0) and 
                    entity.get('end', 0) > existing.get('start', 0)):
                    
                    # Keep the one with higher confidence
                    if entity.get('confidence', 0) > existing.get('confidence', 0):
                        result.remove(existing)
                        result.append(entity)
                    
                    overlapping = True
                    break
            
            if not overlapping:
                result.append(entity)
        
        return result
    
    def _validate_entity(self, entity: Dict, entity_type: str) -> bool:
        """Validate extracted entity"""
        text = entity.get('text', '').strip()
        
        if not text:
            return False
        
        # Type-specific validation
        if entity_type == 'TIME':
            return self._validate_time_entity(text)
        elif entity_type == 'DATE':
            return self._validate_date_entity(text)
        elif entity_type == 'LOCATION':
            return self._is_valid_location(text)
        elif entity_type == 'FARE':
            return self._validate_fare_entity(text)
        
        return True
    
    def _validate_time_entity(self, text: str) -> bool:
        """Validate time entity"""
        time_patterns = [
            r'^\d{1,2}:\d{2}(\s*(AM|PM))?$',
            r'^\d{1,2}\s*(AM|PM)$',
            r'^(morning|afternoon|evening|night)$'
        ]
        
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in time_patterns)
    
    def _validate_date_entity(self, text: str) -> bool:
        """Validate date entity"""
        date_patterns = [
            r'^(today|tomorrow|yesterday)$',
            r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$',
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',
            r'^\d{1,2}-\d{1,2}-\d{2,4}$'
        ]
        
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in date_patterns)
    
    def _validate_fare_entity(self, text: str) -> bool:
        """Validate fare entity"""
        fare_patterns = [
            r'^\d+\s*(rwf|francs?|frw)$',
            r'^(rwf|frw)\s*\d+$'
        ]
        
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in fare_patterns)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the entity extractor"""
        return {
            "initialized": self._initialized,
            "entity_types": list(self.entity_patterns.keys()),
            "custom_entity_types": list(self.custom_entities.keys()),
            "supported_languages": list(self.matchers.keys()),
            "total_patterns": sum(len(patterns) for patterns in self.entity_patterns.values()),
            "custom_entities_count": sum(len(entities) for entities in self.custom_entities.values())
        }
    
    async def retrain_model(self):
        """Retrain entity extraction models (placeholder for future ML-based approach)"""
        ml_logger.info("Entity extractor retraining not implemented (using rule-based approach)")
        pass
    
    def is_initialized(self) -> bool:
        """Check if extractor is initialized"""
        return self._initialized