"""
Intent classification service using machine learning
"""

import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, List, Optional, Any, Tuple
from functools import partial
import asyncio
from datetime import datetime
import json

from config.settings import Settings, INTENT_MAPPINGS
from utils.logger import ml_logger, log_ml_operation
from utils.cache import MLCacheManager

class IntentClassifier:
    """
    Machine learning-based intent classification service
    """
    
    def __init__(self, settings: Settings, cache_manager: Optional[MLCacheManager] = None):
        """
        Initialize intent classifier
        
        Args:
            settings: Application settings
            cache_manager: Cache manager instance
        """
        self.settings = settings
        self.cache_manager = cache_manager
        self.model = None
        self.vectorizer = None
        self.intent_labels = []
        self.confidence_threshold = settings.intent_confidence_threshold
        self._initialized = False
        
        # Training data will be loaded from files or generated
        self.training_data = []
    
    async def initialize(self):
        """Initialize the intent classifier"""
        try:
            start_time = datetime.now()
            ml_logger.info("Initializing intent classifier...")
            
            # Try to load pre-trained model
            if await self._load_model():
                ml_logger.info("Loaded pre-trained intent classifier")
            else:
                # Generate training data and train model
                ml_logger.info("No pre-trained model found, training new model...")
                await self._generate_training_data()
                await self._train_model()
                await self._save_model()
            
            self._initialized = True
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            log_ml_operation("intent_classifier_init", duration, True, {
                "model_type": type(self.model).__name__ if self.model else "None",
                "num_intents": len(self.intent_labels)
            })
            
            ml_logger.info("Intent classifier initialized successfully")
            
        except Exception as e:
            ml_logger.error(f"Failed to initialize intent classifier: {e}")
            log_ml_operation("intent_classifier_init", 0, False, {"error": str(e)})
            raise
    
    async def _load_model(self) -> bool:
        """Load pre-trained model from disk"""
        try:
            model_path = self.settings.intent_model_path
            vectorizer_path = self.settings.intent_vectorizer_path
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                # Load in thread pool to avoid blocking
                self.model = await asyncio.get_event_loop().run_in_executor(
                    None, joblib.load, model_path
                )
                self.vectorizer = await asyncio.get_event_loop().run_in_executor(
                    None, joblib.load, vectorizer_path
                )
                
                # Validate loaded artifacts
                is_valid = True
                # Model must expose classes_ and have at least one class
                if not hasattr(self.model, 'classes_') or not getattr(self.model, 'classes_', []):
                    is_valid = False
                else:
                    self.intent_labels = list(self.model.classes_)
                # Vectorizer must be fitted and have vocabulary
                if not hasattr(self.vectorizer, 'vocabulary_') or not getattr(self.vectorizer, 'vocabulary_', {}):
                    is_valid = False
                # Model should support predict_proba (SVC requires probability=True)
                if not hasattr(self.model, 'predict_proba'):
                    is_valid = False

                if not is_valid:
                    ml_logger.warning("Pre-trained intent artifacts are invalid or unfitted; will retrain.")
                    # Clear loaded artifacts to force training
                    self.model = None
                    self.vectorizer = None
                    self.intent_labels = []
                    return False

                ml_logger.info(f"Loaded model with {len(self.intent_labels)} intent classes")
                return True
            
            return False
            
        except Exception as e:
            ml_logger.warning(f"Failed to load pre-trained model: {e}")
            return False
    
    async def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(os.path.dirname(self.settings.intent_model_path), exist_ok=True)
            
            # Save in thread pool
            await asyncio.get_event_loop().run_in_executor(
                None, joblib.dump, self.model, self.settings.intent_model_path
            )
            await asyncio.get_event_loop().run_in_executor(
                None, joblib.dump, self.vectorizer, self.settings.intent_vectorizer_path
            )
            
            ml_logger.info("Intent classifier model saved successfully")
            
        except Exception as e:
            ml_logger.error(f"Failed to save model: {e}")
    
    async def _generate_training_data(self):
        """Generate training data for intent classification"""
        training_examples = []
        
        # Transport-related examples
        transport_examples = [
            ("What time does the bus to Huye leave?", "transport_query"),
            ("How much is the fare to Musanze?", "transport_query"),
            ("Show me bus schedules from Kigali", "transport_query"),
            ("Is there traffic on the way to Rubavu?", "transport_query"),
            ("Which route goes to Nyagatare?", "transport_query"),
            ("When is the next bus departing?", "transport_query"),
            ("How long does it take to get to Butare?", "transport_query"),
            ("What transport options are available?", "transport_query"),
            ("Are there any delays on Route 1?", "transport_query"),
            ("Book a ticket to Gisenyi", "transport_query"),
            
            # Kinyarwanda transport examples
            ("Bisi igana Huye itaha ryari?", "transport_query"),
            ("Ikiguzi cya Musanze ni angahe?", "transport_query"),
            ("Nyerekana gahunda ya bisi ziva Kigali", "transport_query"),
            ("Hari ubwikorezi mu nzira yerekeza Rubavu?", "transport_query"),
            
            # French transport examples
            ("À quelle heure part le bus pour Huye?", "transport_query"),
            ("Combien coûte le trajet vers Musanze?", "transport_query"),
            ("Montrez-moi les horaires de bus depuis Kigali", "transport_query"),
        ]
        
        # Entertainment examples
        entertainment_examples = [
            ("Tell me a joke", "entertainment_request"),
            ("I want to hear something funny", "entertainment_request"),
            ("Share an interesting fact", "entertainment_request"),
            ("Give me a quiz question", "entertainment_request"),
            ("What's the latest news?", "entertainment_request"),
            ("I'm bored, entertain me", "entertainment_request"),
            ("Tell me a story", "entertainment_request"),
            ("What's happening in the world?", "entertainment_request"),
            ("I need some motivation", "entertainment_request"),
            ("Share a fun fact about Rwanda", "entertainment_request"),
            
            # Kinyarwanda entertainment
            ("Mpa urwenya", "entertainment_request"),
            ("Ndashaka kwinezeza", "entertainment_request"),
            ("Mpa amakuru mashya", "entertainment_request"),
            
            # French entertainment
            ("Racontez-moi une blague", "entertainment_request"),
            ("Je veux m'amuser", "entertainment_request"),
            ("Partagez un fait intéressant", "entertainment_request"),
        ]
        
        # Support examples
        support_examples = [
            ("I need help", "support_request"),
            ("How do I contact support?", "support_request"),
            ("I have a problem", "support_request"),
            ("Something is not working", "support_request"),
            ("Can you assist me?", "support_request"),
            ("I have a question about the app", "support_request"),
            ("Report a bug", "support_request"),
            ("I need technical assistance", "support_request"),
            ("How do I reset my password?", "support_request"),
            ("Contact customer service", "support_request"),
            
            # Kinyarwanda support
            ("Nkeneye ubufasha", "support_request"),
            ("Mfite ikibazo", "support_request"),
            ("Ntabwo birakora neza", "support_request"),
            
            # French support
            ("J'ai besoin d'aide", "support_request"),
            ("J'ai un problème", "support_request"),
            ("Contactez le support", "support_request"),
        ]
        
        # Greeting examples
        greeting_examples = [
            ("Hello", "greeting"),
            ("Hi there", "greeting"),
            ("Good morning", "greeting"),
            ("Good afternoon", "greeting"),
            ("Hey", "greeting"),
            ("Greetings", "greeting"),
            ("How are you?", "greeting"),
            ("Nice to meet you", "greeting"),
            
            # Kinyarwanda greetings
            ("Muraho", "greeting"),
            ("Mwaramutse", "greeting"),
            ("Mwiriwe", "greeting"),
            ("Bite", "greeting"),
            ("Amakuru", "greeting"),
            
            # French greetings
            ("Bonjour", "greeting"),
            ("Salut", "greeting"),
            ("Bonsoir", "greeting"),
            ("Comment allez-vous?", "greeting"),
        ]
        
        # Goodbye examples
        goodbye_examples = [
            ("Goodbye", "goodbye"),
            ("See you later", "goodbye"),
            ("Bye", "goodbye"),
            ("Take care", "goodbye"),
            ("Until next time", "goodbye"),
            ("Farewell", "goodbye"),
            ("Have a good day", "goodbye"),
            
            # Kinyarwanda goodbyes
            ("Murabeho", "goodbye"),
            ("Tugire amahoro", "goodbye"),
            ("Tuzabonana", "goodbye"),
            
            # French goodbyes
            ("Au revoir", "goodbye"),
            ("À bientôt", "goodbye"),
            ("Bonne journée", "goodbye"),
        ]
        
        # General examples
        general_examples = [
            ("What can you do?", "general"),
            ("How does this work?", "general"),
            ("Tell me about yourself", "general"),
            ("What services do you offer?", "general"),
            ("I don't understand", "general"),
            ("Can you help me with something?", "general"),
            ("What is this app about?", "general"),
            ("Explain this feature", "general"),
        ]
        
        # Combine all examples
        all_examples = (
            transport_examples + entertainment_examples + support_examples +
            greeting_examples + goodbye_examples + general_examples
        )
        
        # Add to training data
        for text, intent in all_examples:
            training_examples.append({
                "text": text.lower().strip(),
                "intent": intent
            })
        
        # Add keyword-based examples for each language
        for lang, intent_keywords in INTENT_MAPPINGS.items():
            for intent, keywords in intent_keywords.items():
                for keyword in keywords:
                    training_examples.append({
                        "text": keyword,
                        "intent": intent
                    })
        
        self.training_data = training_examples
        self.intent_labels = list(set(example["intent"] for example in training_examples))
        
        ml_logger.info(f"Generated {len(training_examples)} training examples for {len(self.intent_labels)} intents")
    
    async def _train_model(self):
        """Train the intent classification model"""
        if not self.training_data:
            raise ValueError("No training data available")
        
        start_time = datetime.now()
        
        # Prepare training data
        texts = [example["text"] for example in self.training_data]
        labels = [example["intent"] for example in self.training_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=self.settings.training_config["intent_classifier"]["test_size"],
            random_state=self.settings.training_config["intent_classifier"]["random_state"],
            stratify=labels
        )
        
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Vectorize training data
        X_train_vec = await asyncio.get_event_loop().run_in_executor(
            None, self.vectorizer.fit_transform, X_train
        )
        
        # Choose and train model based on configuration
        algorithm = self.settings.training_config["intent_classifier"]["algorithm"]
        
        if algorithm == "svm":
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        elif algorithm == "naive_bayes":
            self.model = MultinomialNB()
        elif algorithm == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Train model
        await asyncio.get_event_loop().run_in_executor(
            None, self.model.fit, X_train_vec, y_train
        )
        
        # Evaluate model
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation (use kwargs to match signature in executor)
        cv_func = partial(
            cross_val_score,
            self.model,
            X_train_vec,
            y_train,
            cv=self.settings.training_config["intent_classifier"]["cross_validation_folds"],
            scoring=None,
        )
        cv_scores = await asyncio.get_event_loop().run_in_executor(None, cv_func)
        
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        ml_logger.log_training_complete(
            "intent_classifier",
            duration / 1000,
            {
                "accuracy": accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "training_size": len(X_train),
                "test_size": len(X_test),
                "num_intents": len(self.intent_labels)
            }
        )
        
        log_ml_operation("intent_classifier_training", duration, True, {
            "accuracy": accuracy,
            "algorithm": algorithm,
            "training_samples": len(X_train)
        })
    
    async def classify_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify intent of input text
        
        Args:
            text: Input text to classify
            context: Additional context information
            
        Returns:
            Dictionary with intent, confidence, and alternatives
        """
        if not self._initialized or not self.model or not self.vectorizer:
            raise ValueError("Intent classifier not initialized")
        
        if not text or not text.strip():
            return {
                "intent": "general",
                "confidence": 0.0,
                "alternatives": []
            }
        
        try:
            start_time = datetime.now()
            
            # Check cache first
            if self.cache_manager:
                cached_result = await self.cache_manager.get_cached_intent(text, context)
                if cached_result:
                    return cached_result
            
            # Preprocess text
            processed_text = text.lower().strip()
            
            # Vectorize input
            text_vec = self.vectorizer.transform([processed_text])
            
            # Get prediction probabilities
            probabilities = await asyncio.get_event_loop().run_in_executor(
                None, self.model.predict_proba, text_vec
            )
            
            # Get class probabilities
            class_probs = probabilities[0]
            intent_probs = list(zip(self.model.classes_, class_probs))
            intent_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Primary prediction
            predicted_intent = intent_probs[0][0]
            confidence = float(intent_probs[0][1])
            
            # Alternative predictions
            alternatives = [
                {"intent": intent, "confidence": float(prob)}
                for intent, prob in intent_probs[1:5]  # Top 4 alternatives
                if prob > 0.1  # Only include reasonable alternatives
            ]
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                # Check if it's a keyword match
                keyword_intent = self._check_keyword_match(processed_text)
                if keyword_intent:
                    predicted_intent = keyword_intent
                    confidence = max(confidence, 0.6)  # Boost confidence for keyword matches
                else:
                    predicted_intent = "general"
            
            result = {
                "intent": predicted_intent,
                "confidence": confidence,
                "alternatives": alternatives
            }
            
            # Cache result
            if self.cache_manager:
                await self.cache_manager.cache_intent_prediction(
                    text, predicted_intent, confidence, context
                )
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log_ml_operation("intent_classification", duration, True, {
                "intent": predicted_intent,
                "confidence": confidence,
                "text_length": len(text)
            })
            
            return result
            
        except Exception as e:
            ml_logger.error(f"Intent classification error: {e}")
            log_ml_operation("intent_classification", 0, False, {"error": str(e)})
            
            return {
                "intent": "general",
                "confidence": 0.0,
                "alternatives": []
            }
    
    def _check_keyword_match(self, text: str) -> Optional[str]:
        """Check for direct keyword matches"""
        text_lower = text.lower()
        
        # Check each intent's keywords
        for lang, intent_keywords in INTENT_MAPPINGS.items():
            for intent, keywords in intent_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        return intent
        
        return None
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the intent classifier model"""
        return {
            "initialized": self._initialized,
            "model_type": type(self.model).__name__ if self.model else None,
            "num_intents": len(self.intent_labels),
            "intent_labels": self.intent_labels,
            "confidence_threshold": self.confidence_threshold,
            "training_samples": len(self.training_data) if self.training_data else 0,
            "vectorizer_features": self.vectorizer.get_feature_names_out().shape[0] if self.vectorizer else 0
        }
    
    async def retrain_model(self):
        """Retrain the model with current data"""
        try:
            ml_logger.info("Starting intent classifier retraining...")
            
            if not self.training_data:
                await self._generate_training_data()
            
            await self._train_model()
            await self._save_model()
            
            ml_logger.info("Intent classifier retraining completed")
            
        except Exception as e:
            ml_logger.error(f"Model retraining failed: {e}")
            raise
    
    def is_initialized(self) -> bool:
        """Check if classifier is initialized"""
        return self._initialized