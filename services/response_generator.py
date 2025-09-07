"""
Response generation service for creating contextual AI responses
"""

import json
import os
import random
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import httpx

from config.settings import Settings, RESPONSE_TEMPLATES
from utils.logger import ml_logger, log_ml_operation
from utils.cache import MLCacheManager

class ResponseGenerator:
    """
    Service for generating contextual responses based on intent and entities
    """
    
    def __init__(self, settings: Settings, cache_manager: Optional[MLCacheManager] = None):
        """
        Initialize response generator
        
        Args:
            settings: Application settings
            cache_manager: Cache manager instance
        """
        self.settings = settings
        self.cache_manager = cache_manager
        self.response_templates = {}
        self.dynamic_responses = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the response generator"""
        try:
            start_time = datetime.now()
            ml_logger.info("Initializing response generator...")
            
            # Load response templates
            await self._load_response_templates()
            
            # Initialize dynamic response generators
            self._initialize_dynamic_generators()
            
            self._initialized = True
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            log_ml_operation("response_generator_init", duration, True, {
                "template_languages": len(self.response_templates),
                "dynamic_generators": len(self.dynamic_responses)
            })
            
            ml_logger.info("Response generator initialized successfully")
            
        except Exception as e:
            ml_logger.error(f"Failed to initialize response generator: {e}")
            log_ml_operation("response_generator_init", 0, False, {"error": str(e)})
            raise
    
    async def _load_response_templates(self):
        """Load response templates from configuration"""
        try:
            # Try to load from file first
            if os.path.exists(self.settings.response_templates_path):
                with open(self.settings.response_templates_path, 'r', encoding='utf-8') as f:
                    self.response_templates = json.load(f)
                ml_logger.info("Loaded response templates from file")
            else:
                # Use default templates
                self.response_templates = RESPONSE_TEMPLATES
                await self._save_response_templates()
                ml_logger.info("Using default response templates")
            
        except Exception as e:
            ml_logger.warning(f"Failed to load response templates: {e}")
            self.response_templates = RESPONSE_TEMPLATES
    
    async def _save_response_templates(self):
        """Save response templates to file"""
        try:
            os.makedirs(os.path.dirname(self.settings.response_templates_path), exist_ok=True)
            with open(self.settings.response_templates_path, 'w', encoding='utf-8') as f:
                json.dump(self.response_templates, f, indent=2, ensure_ascii=False)
        except Exception as e:
            ml_logger.warning(f"Failed to save response templates: {e}")
    
    def _initialize_dynamic_generators(self):
        """Initialize dynamic response generators for different intents"""
        self.dynamic_responses = {
            'transport_query': self._generate_transport_response,
            'entertainment_request': self._generate_entertainment_response,
            'support_request': self._generate_support_response,
            'greeting': self._generate_greeting_response,
            'goodbye': self._generate_goodbye_response,
            'general': self._generate_general_response
        }
    
    async def generate_response(
        self,
        intent: str,
        entities: Dict[str, Any],
        original_message: str,
        user_context: Optional[Dict[str, Any]] = None,
        language: str = "en",
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate contextual response based on intent and entities
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            original_message: Original user message
            user_context: User context information
            language: Response language
            confidence: Intent confidence score
            
        Returns:
            Dictionary with response and metadata
        """
        if not self._initialized:
            raise ValueError("Response generator not initialized")
        
        try:
            start_time = datetime.now()
            
            # Check cache first
            cache_key = self._generate_cache_key(intent, entities, original_message, language)
            if self.cache_manager:
                cached_response = await self.cache_manager.get(cache_key)
                if cached_response:
                    return cached_response
            
            # Generate response based on intent
            if intent in self.dynamic_responses:
                response_data = await self.dynamic_responses[intent](
                    entities, original_message, user_context, language, confidence
                )
            else:
                # Fallback to general response
                response_data = await self._generate_general_response(
                    entities, original_message, user_context, language, confidence
                )
            
            # Add metadata
            response_data['metadata'] = {
                **response_data.get('metadata', {}),
                'intent': intent,
                'confidence': confidence,
                'language': language,
                'generation_method': response_data.get('generation_method', 'template'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the response
            if self.cache_manager:
                await self.cache_manager.set(
                    cache_key, 
                    response_data, 
                    ttl=self.settings.cache_config.get('response_generation', 600)
                )
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log_ml_operation("response_generation", duration, True, {
                "intent": intent,
                "language": language,
                "entities_count": len(entities),
                "method": response_data.get('generation_method', 'template')
            })
            
            return response_data
            
        except Exception as e:
            ml_logger.error(f"Response generation error: {e}")
            log_ml_operation("response_generation", 0, False, {"error": str(e)})
            
            # Return fallback response
            return {
                "response": self._get_fallback_response(language),
                "needs_translation": False,
                "metadata": {
                    "fallback": True,
                    "error": str(e)
                }
            }
    
    async def _generate_transport_response(
        self,
        entities: Dict[str, Any],
        message: str,
        context: Optional[Dict[str, Any]],
        language: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Generate response for transport queries"""
        
        # Extract relevant entities
        origin = self._extract_entity_value(entities, 'ORIGIN')
        destination = self._extract_entity_value(entities, 'DESTINATION')
        location = self._extract_entity_value(entities, 'LOCATION')
        time = self._extract_entity_value(entities, 'TIME')
        transport_type = self._extract_entity_value(entities, 'TRANSPORT_TYPE')
        
        # Determine specific transport query type
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['schedule', 'time', 'when', 'departure', 'arrival']):
            return await self._generate_schedule_response(
                origin, destination, location, time, transport_type, language
            )
        elif any(word in message_lower for word in ['fare', 'cost', 'price', 'how much']):
            return await self._generate_fare_response(
                origin, destination, location, transport_type, language
            )
        elif any(word in message_lower for word in ['traffic', 'jam', 'delay', 'congestion']):
            return await self._generate_traffic_response(location, language)
        elif any(word in message_lower for word in ['route', 'way', 'path', 'direction']):
            return await self._generate_route_response(
                origin, destination, location, transport_type, language
            )
        else:
            # General transport query
            return await self._generate_general_transport_response(
                origin, destination, location, transport_type, language
            )
    
    async def _generate_schedule_response(
        self,
        origin: Optional[str],
        destination: Optional[str],
        location: Optional[str],
        time: Optional[str],
        transport_type: Optional[str],
        language: str
    ) -> Dict[str, Any]:
        """Generate schedule-specific response"""
        
        # Mock schedule data (in production, this would query the transport API)
        schedules = {
            "kigali-huye": ["06:00 AM", "08:00 AM", "10:00 AM", "02:00 PM", "04:00 PM", "06:00 PM"],
            "kigali-musanze": ["07:00 AM", "09:00 AM", "11:00 AM", "01:00 PM", "03:00 PM", "05:00 PM"],
            "kigali-rubavu": ["06:30 AM", "10:00 AM", "02:00 PM", "05:30 PM"]
        }
        
        response_templates = {
            "en": {
                "specific_route": "The {transport_type} from {origin} to {destination} departs at: {schedule}",
                "general_location": "Here are the {transport_type} schedules for {location}: {schedule}",
                "no_route": "I couldn't find specific schedule information for that route. Please check with the transport operator or try a different route.",
                "general": "Bus schedules typically run from 6:00 AM to 7:00 PM. For specific times, please specify your origin and destination."
            },
            "rw": {
                "specific_route": "{transport_type} iva {origin} yerekeza {destination} itaha: {schedule}",
                "general_location": "Dore gahunda ya {transport_type} ya {location}: {schedule}",
                "no_route": "Sinabonye amakuru yihariye yerekeye iyo nzira. Nyabuneka raba n'uwuyikoresha cyangwa ugerageze indi nzira.",
                "general": "Gahunda ya bisi mubisanzwe itangira saa kumi n'ebyiri z'igitondo kugeza saa imwe z'ijoro. Kugirango ubone ibihe byihariye, nyabuneka sobanura aho uva n'aho ugiye."
            },
            "fr": {
                "specific_route": "Le {transport_type} de {origin} à {destination} part à: {schedule}",
                "general_location": "Voici les horaires de {transport_type} pour {location}: {schedule}",
                "no_route": "Je n'ai pas trouvé d'informations d'horaire spécifiques pour cette route. Veuillez vérifier avec l'opérateur de transport ou essayer une autre route.",
                "general": "Les horaires de bus fonctionnent généralement de 6h00 à 19h00. Pour des heures spécifiques, veuillez préciser votre origine et destination."
            }
        }
        
        templates = response_templates.get(language, response_templates["en"])
        transport_type = transport_type or "bus"
        
        # Try to find matching schedule
        if origin and destination:
            route_key = f"{origin.lower()}-{destination.lower()}"
            if route_key in schedules:
                schedule_list = schedules[route_key]
                schedule_text = ", ".join(schedule_list)
                response = templates["specific_route"].format(
                    transport_type=transport_type,
                    origin=origin,
                    destination=destination,
                    schedule=schedule_text
                )
            else:
                response = templates["no_route"]
        elif location:
            # Mock schedule for location
            schedule_text = "06:00 AM, 08:00 AM, 10:00 AM, 02:00 PM, 04:00 PM, 06:00 PM"
            response = templates["general_location"].format(
                transport_type=transport_type,
                location=location,
                schedule=schedule_text
            )
        else:
            response = templates["general"]
        
        return {
            "response": response,
            "needs_translation": False,
            "generation_method": "dynamic_transport_schedule",
            "metadata": {
                "query_type": "schedule",
                "origin": origin,
                "destination": destination,
                "location": location,
                "transport_type": transport_type
            }
        }
    
    async def _generate_fare_response(
        self,
        origin: Optional[str],
        destination: Optional[str],
        location: Optional[str],
        transport_type: Optional[str],
        language: str
    ) -> Dict[str, Any]:
        """Generate fare-specific response"""
        
        # Mock fare data
        fares = {
            "kigali-huye": "2,500 RWF",
            "kigali-musanze": "2,000 RWF",
            "kigali-rubavu": "3,000 RWF",
            "kigali-nyagatare": "2,200 RWF"
        }
        
        response_templates = {
            "en": {
                "specific_route": "The fare for {transport_type} from {origin} to {destination} is {fare}.",
                "no_route": "I don't have specific fare information for that route. Typical bus fares range from 1,500 to 3,500 RWF depending on distance.",
                "general": "Bus fares in Rwanda typically range from 500 RWF for short distances to 3,500 RWF for longer routes. Please specify your route for exact pricing."
            },
            "rw": {
                "specific_route": "Ikiguzi cya {transport_type} kuva {origin} kugeza {destination} ni {fare}.",
                "no_route": "Nta makuru mfite yerekeye ikiguzi cy'iyo nzira. Mubisanzwe ikiguzi cya bisi kiva kuri 1,500 kugeza 3,500 RWF bitewe n'intera.",
                "general": "Ikiguzi cya bisi mu Rwanda mubisanzwe kiva kuri 500 RWF ku ntera ngufi kugeza 3,500 RWF ku ntera ndende. Nyabuneka sobanura inzira yawe kugirango ubone ikiguzi nyacyo."
            },
            "fr": {
                "specific_route": "Le tarif pour {transport_type} de {origin} à {destination} est {fare}.",
                "no_route": "Je n'ai pas d'informations tarifaires spécifiques pour cette route. Les tarifs de bus typiques varient de 1 500 à 3 500 RWF selon la distance.",
                "general": "Les tarifs de bus au Rwanda varient généralement de 500 RWF pour les courtes distances à 3 500 RWF pour les routes plus longues. Veuillez spécifier votre itinéraire pour un prix exact."
            }
        }
        
        templates = response_templates.get(language, response_templates["en"])
        transport_type = transport_type or "bus"
        
        if origin and destination:
            route_key = f"{origin.lower()}-{destination.lower()}"
            if route_key in fares:
                fare = fares[route_key]
                response = templates["specific_route"].format(
                    transport_type=transport_type,
                    origin=origin,
                    destination=destination,
                    fare=fare
                )
            else:
                response = templates["no_route"]
        else:
            response = templates["general"]
        
        return {
            "response": response,
            "needs_translation": False,
            "generation_method": "dynamic_transport_fare",
            "metadata": {
                "query_type": "fare",
                "origin": origin,
                "destination": destination,
                "transport_type": transport_type
            }
        }
    
    async def _generate_traffic_response(
        self,
        location: Optional[str],
        language: str
    ) -> Dict[str, Any]:
        """Generate traffic information response"""
        
        # Mock traffic data
        traffic_conditions = ["light", "moderate", "heavy"]
        current_condition = random.choice(traffic_conditions)
        
        response_templates = {
            "en": {
                "specific_location": f"Current traffic conditions {f'in {location}' if location else 'on major routes'} are {current_condition}. {self._get_traffic_advice(current_condition, language)}",
                "general": f"Current traffic conditions on major routes are {current_condition}. {self._get_traffic_advice(current_condition, language)}"
            },
            "rw": {
                "specific_location": f"Ubwikorezi {f'muri {location}' if location else 'mu nzira nyamukuru'} ubu ni {self._translate_traffic_condition(current_condition, 'rw')}. {self._get_traffic_advice(current_condition, 'rw')}",
                "general": f"Ubwikorezi mu nzira nyamukuru ubu ni {self._translate_traffic_condition(current_condition, 'rw')}. {self._get_traffic_advice(current_condition, 'rw')}"
            },
            "fr": {
                "specific_location": f"Les conditions de circulation actuelles {f'à {location}' if location else 'sur les routes principales'} sont {self._translate_traffic_condition(current_condition, 'fr')}. {self._get_traffic_advice(current_condition, 'fr')}",
                "general": f"Les conditions de circulation actuelles sur les routes principales sont {self._translate_traffic_condition(current_condition, 'fr')}. {self._get_traffic_advice(current_condition, 'fr')}"
            }
        }
        
        templates = response_templates.get(language, response_templates["en"])
        
        if location:
            response = templates["specific_location"]
        else:
            response = templates["general"]
        
        return {
            "response": response,
            "needs_translation": False,
            "generation_method": "dynamic_traffic",
            "metadata": {
                "query_type": "traffic",
                "location": location,
                "traffic_condition": current_condition
            }
        }
    
    async def _generate_entertainment_response(
        self,
        entities: Dict[str, Any],
        message: str,
        context: Optional[Dict[str, Any]],
        language: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Generate response for entertainment requests"""
        
        entertainment_type = self._extract_entity_value(entities, 'ENTERTAINMENT_TYPE')
        message_lower = message.lower()
        
        # Determine entertainment type from message if not extracted
        if not entertainment_type:
            if any(word in message_lower for word in ['joke', 'funny', 'humor', 'laugh']):
                entertainment_type = 'joke'
            elif any(word in message_lower for word in ['fact', 'information', 'learn', 'knowledge']):
                entertainment_type = 'fact'
            elif any(word in message_lower for word in ['quiz', 'question', 'test', 'challenge']):
                entertainment_type = 'quiz'
            elif any(word in message_lower for word in ['news', 'updates', 'latest', 'current']):
                entertainment_type = 'news'
            elif any(word in message_lower for word in ['story', 'tale', 'narrative']):
                entertainment_type = 'story'
            else:
                entertainment_type = 'general'
        
        response_templates = {
            "en": {
                "redirect": "I'd be happy to entertain you! Let me fetch some {entertainment_type} content for you. You can also ask for jokes, facts, quiz questions, or the latest news.",
                "general": "I can entertain you with jokes, interesting facts, quiz questions, latest news, or motivational content. What would you prefer?"
            },
            "rw": {
                "redirect": "Nishimiye kugukeza! Reka nkugendere {entertainment_type}. Ushobora kandi gusaba urwenya, amakuru mashya, ibibazo by'ikizamini, cyangwa amakuru y'ubu.",
                "general": "Nshobora kugukeza n'urwenya, amakuru ashimishije, ibibazo by'ikizamini, amakuru mashya, cyangwa ubutumwa bwishimangira. Ni iki ushaka?"
            },
            "fr": {
                "redirect": "Je serais ravi de vous divertir! Laissez-moi vous chercher du contenu {entertainment_type}. Vous pouvez aussi demander des blagues, des faits, des questions de quiz, ou les dernières nouvelles.",
                "general": "Je peux vous divertir avec des blagues, des faits intéressants, des questions de quiz, les dernières nouvelles, ou du contenu motivationnel. Que préférez-vous?"
            }
        }
        
        templates = response_templates.get(language, response_templates["en"])
        
        if entertainment_type and entertainment_type != 'general':
            response = templates["redirect"].format(entertainment_type=entertainment_type)
        else:
            response = templates["general"]
        
        return {
            "response": response,
            "needs_translation": False,
            "generation_method": "dynamic_entertainment",
            "metadata": {
                "entertainment_type": entertainment_type,
                "redirect_to_api": True
            }
        }
    
    async def _generate_support_response(
        self,
        entities: Dict[str, Any],
        message: str,
        context: Optional[Dict[str, Any]],
        language: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Generate response for support requests"""
        
        message_lower = message.lower()
        
        # Determine support type
        if any(word in message_lower for word in ['bug', 'error', 'broken', 'not working']):
            support_type = 'technical'
        elif any(word in message_lower for word in ['account', 'login', 'password', 'profile']):
            support_type = 'account'
        elif any(word in message_lower for word in ['how to', 'help with', 'tutorial', 'guide']):
            support_type = 'how_to'
        else:
            support_type = 'general'
        
        response_templates = {
            "en": {
                "technical": "I understand you're experiencing a technical issue. I can help you troubleshoot or you can contact our technical support team. Can you describe the specific problem you're facing?",
                "account": "I can help you with account-related questions. For security reasons, some account changes may require verification. What specific account issue can I assist you with?",
                "how_to": "I'm here to help you learn how to use our services. What specific feature or process would you like guidance on?",
                "general": "I'm here to help! You can ask me about transport schedules, entertainment content, or if you need technical support. What can I assist you with today?"
            },
            "rw": {
                "technical": "Ndumva ufite ikibazo cy'ubuhanga. Nshobora kugufasha gukemura ikibazo cyangwa uvugane n'itsinda ryacu ry'ubufasha bw'ubuhanga. Ushobora kusobanura neza ikibazo ugenda guhura nacyo?",
                "account": "Nshobora kugufasha mu bibazo bijyanye na konti yawe. Kubw'umutekano, zimwe mu mpinduka za konti zishobora gusaba kwemeza. Ni ikihe kibazo cya konti nkugufashaho?",
                "how_to": "Ndi hano kugufasha kwiga uko ukoresha serivisi zacu. Ni iyihe nteruro cyangwa inzira ushaka kuyobora?",
                "general": "Ndi hano kugufasha! Ushobora kumbaza ku gahunda y'ingendo, ibintu by'kwishimisha, cyangwa niba ukeneye ubufasha bw'ubuhanga. Ni iki nkugufashaho uyu munsi?"
            },
            "fr": {
                "technical": "Je comprends que vous rencontrez un problème technique. Je peux vous aider à résoudre le problème ou vous pouvez contacter notre équipe de support technique. Pouvez-vous décrire le problème spécifique que vous rencontrez?",
                "account": "Je peux vous aider avec les questions liées au compte. Pour des raisons de sécurité, certaines modifications de compte peuvent nécessiter une vérification. Avec quel problème de compte spécifique puis-je vous aider?",
                "how_to": "Je suis là pour vous aider à apprendre comment utiliser nos services. Sur quelle fonctionnalité ou processus spécifique aimeriez-vous des conseils?",
                "general": "Je suis là pour vous aider! Vous pouvez me demander des horaires de transport, du contenu de divertissement, ou si vous avez besoin d'un support technique. Avec quoi puis-je vous aider aujourd'hui?"
            }
        }
        
        templates = response_templates.get(language, response_templates["en"])
        response = templates.get(support_type, templates["general"])
        
        return {
            "response": response,
            "needs_translation": False,
            "generation_method": "dynamic_support",
            "metadata": {
                "support_type": support_type,
                "escalation_available": True
            }
        }
    
    async def _generate_greeting_response(
        self,
        entities: Dict[str, Any],
        message: str,
        context: Optional[Dict[str, Any]],
        language: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Generate greeting response"""
        
        templates = self.response_templates.get(language, {}).get('greeting', [])
        if not templates:
            templates = self.response_templates['en']['greeting']
        
        response = random.choice(templates)
        
        # Add personalization if user context is available
        if context and context.get('user', {}).get('firstName'):
            name = context['user']['firstName']
            personalized_templates = {
                "en": f"Hello {name}! How can I help you today?",
                "rw": f"Muraho {name}! Ni gute nkugufasha uyu munsi?",
                "fr": f"Bonjour {name}! Comment puis-je vous aider aujourd'hui?"
            }
            response = personalized_templates.get(language, response)
        
        return {
            "response": response,
            "needs_translation": False,
            "generation_method": "template_greeting",
            "metadata": {
                "personalized": bool(context and context.get('user', {}).get('firstName'))
            }
        }
    
    async def _generate_goodbye_response(
        self,
        entities: Dict[str, Any],
        message: str,
        context: Optional[Dict[str, Any]],
        language: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Generate goodbye response"""
        
        templates = self.response_templates.get(language, {}).get('goodbye', [])
        if not templates:
            templates = self.response_templates['en']['goodbye']
        
        response = random.choice(templates)
        
        return {
            "response": response,
            "needs_translation": False,
            "generation_method": "template_goodbye",
            "metadata": {}
        }
    
    async def _generate_general_response(
        self,
        entities: Dict[str, Any],
        message: str,
        context: Optional[Dict[str, Any]],
        language: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Generate general response"""
        
        templates = self.response_templates.get(language, {}).get('general', [])
        if not templates:
            templates = self.response_templates['en']['general']
        
        response = random.choice(templates)
        
        return {
            "response": response,
            "needs_translation": False,
            "generation_method": "template_general",
            "metadata": {
                "low_confidence": confidence < 0.5
            }
        }
    
    def _extract_entity_value(self, entities: Dict[str, Any], entity_type: str) -> Optional[str]:
        """Extract entity value from entities dictionary"""
        entity_list = entities.get(entity_type, [])
        if entity_list and isinstance(entity_list, list) and len(entity_list) > 0:
            return entity_list[0].get('text', '').strip()
        return None
    
    def _generate_cache_key(
        self,
        intent: str,
        entities: Dict[str, Any],
        message: str,
        language: str
    ) -> str:
        """Generate cache key for response"""
        # Create a simplified key based on intent, main entities, and language
        key_parts = [f"response:{intent}:{language}"]
        
        # Add main entity values
        main_entities = ['ORIGIN', 'DESTINATION', 'LOCATION', 'ENTERTAINMENT_TYPE']
        for entity_type in main_entities:
            value = self._extract_entity_value(entities, entity_type)
            if value:
                key_parts.append(f"{entity_type.lower()}:{value.lower()}")
        
        return ":".join(key_parts)
    
    def _get_fallback_response(self, language: str) -> str:
        """Get fallback response when generation fails"""
        fallback_responses = {
            "en": "I'm sorry, I'm having trouble processing your request right now. Please try again or contact support if the issue persists.",
            "rw": "Ihangane, nfite ikibazo mu gukora icyo usaba. Nyabuneka ongere ugerageze cyangwa uvugane n'abafasha niba ikibazo gikomeje.",
            "fr": "Je suis désolé, j'ai des difficultés à traiter votre demande en ce moment. Veuillez réessayer ou contacter le support si le problème persiste."
        }
        return fallback_responses.get(language, fallback_responses["en"])
    
    def _get_traffic_advice(self, condition: str, language: str) -> str:
        """Get traffic advice based on condition"""
        advice = {
            "en": {
                "light": "Good time to travel!",
                "moderate": "Allow extra time for your journey.",
                "heavy": "Consider alternative routes or delay your trip if possible."
            },
            "rw": {
                "light": "Ni igihe cyiza cyo kugenda!",
                "moderate": "Tanga igihe cyongeyeho ku rugendo rwawe.",
                "heavy": "Tekereza ku zindi nzira cyangwa utinde urugendo rwawe niba bishoboka."
            },
            "fr": {
                "light": "Bon moment pour voyager!",
                "moderate": "Prévoyez du temps supplémentaire pour votre voyage.",
                "heavy": "Considérez des itinéraires alternatifs ou retardez votre voyage si possible."
            }
        }
        return advice.get(language, advice["en"]).get(condition, "")
    
    def _translate_traffic_condition(self, condition: str, language: str) -> str:
        """Translate traffic condition to target language"""
        translations = {
            "rw": {
                "light": "buke",
                "moderate": "bwagereranije",
                "heavy": "bwinshi"
            },
            "fr": {
                "light": "fluide",
                "moderate": "modérée",
                "heavy": "dense"
            }
        }
        return translations.get(language, {}).get(condition, condition)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the response generator"""
        return {
            "initialized": self._initialized,
            "supported_languages": list(self.response_templates.keys()),
            "supported_intents": list(self.dynamic_responses.keys()),
            "template_count": sum(
                len(intent_templates) 
                for lang_templates in self.response_templates.values()
                for intent_templates in lang_templates.values()
            )
        }
    
    def is_initialized(self) -> bool:
        """Check if generator is initialized"""
        return self._initialized