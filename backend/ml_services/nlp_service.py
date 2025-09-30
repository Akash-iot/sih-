"""
NLP and PII Extraction Service for ETHEREYE
Implements advanced NLP capabilities using spaCy, NLTK, and HuggingFace Transformers
for blockchain-related text analysis and PII detection
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime
import numpy as np
import pandas as pd

# NLP Libraries
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, AutoModelForTokenClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers")

# Configure logging
logger = logging.getLogger(__name__)

class BlockchainNLPService:
    """
    Advanced NLP service for blockchain-related text analysis
    Combines spaCy, NLTK, and HuggingFace for comprehensive text processing
    """
    
    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.classification_pipeline = None
        
        # Initialize models
        self._initialize_models()
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?\d{1,4}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'bitcoin_address': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            'ethereum_address': r'\b0x[a-fA-F0-9]{40}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        }
        
        # Blockchain-specific keywords
        self.blockchain_keywords = {
            'wallet_terms': ['wallet', 'address', 'private key', 'seed phrase', 'mnemonic'],
            'transaction_terms': ['transfer', 'send', 'receive', 'transaction', 'tx', 'hash'],
            'exchange_terms': ['exchange', 'binance', 'coinbase', 'kraken', 'huobi'],
            'defi_terms': ['defi', 'uniswap', 'compound', 'aave', 'sushiswap'],
            'suspicious_terms': ['mixer', 'tumbler', 'privacy coin', 'monero', 'zcash'],
            'scam_terms': ['phishing', 'fake', 'scam', 'fraud', 'ponzi', 'rug pull']
        }
    
    def _initialize_models(self):
        """Initialize NLP models and pipelines"""
        try:
            # Initialize spaCy
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded successfully")
                except OSError:
                    logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            
            # Initialize NLTK
            if NLTK_AVAILABLE:
                try:
                    # Download required NLTK data
                    nltk.download('vader_lexicon', quiet=True)
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                    nltk.download('maxent_ne_chunker', quiet=True)
                    nltk.download('words', quiet=True)
                    
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    logger.info("NLTK models initialized successfully")
                except Exception as e:
                    logger.warning(f"NLTK initialization failed: {e}")
            
            # Initialize HuggingFace pipelines
            if TRANSFORMERS_AVAILABLE:
                try:
                    # NER pipeline for PII detection
                    self.ner_pipeline = pipeline(
                        "ner",
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        aggregation_strategy="simple"
                    )
                    
                    # Classification pipeline for sentiment/content analysis
                    self.classification_pipeline = pipeline(
                        "text-classification",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                    )
                    
                    logger.info("HuggingFace pipelines initialized successfully")
                except Exception as e:
                    logger.warning(f"HuggingFace initialization failed: {e}")
                    
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
    
    def extract_pii_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract PII (Personally Identifiable Information) from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing PII extraction results
        """
        try:
            pii_found = {}
            confidence_scores = {}
            
            # Pattern-based PII extraction
            for pii_type, pattern in self.pii_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    pii_found[pii_type] = {
                        'matches': list(set(matches)),  # Remove duplicates
                        'count': len(matches),
                        'method': 'regex'
                    }
                    confidence_scores[pii_type] = 0.9  # High confidence for regex matches
            
            # spaCy NER extraction
            spacy_entities = self._extract_spacy_entities(text)
            if spacy_entities:
                for entity_type, entities in spacy_entities.items():
                    if entity_type not in pii_found:
                        pii_found[entity_type] = {
                            'matches': entities['matches'],
                            'count': entities['count'],
                            'method': 'spacy'
                        }
                        confidence_scores[entity_type] = entities.get('confidence', 0.7)
            
            # HuggingFace NER extraction
            hf_entities = self._extract_huggingface_entities(text)
            if hf_entities:
                for entity_type, entities in hf_entities.items():
                    if entity_type not in pii_found:
                        pii_found[entity_type] = {
                            'matches': entities['matches'],
                            'count': entities['count'],
                            'method': 'transformers'
                        }
                        confidence_scores[entity_type] = entities.get('confidence', 0.8)
            
            # Calculate overall PII risk score
            risk_score = self._calculate_pii_risk_score(pii_found, confidence_scores)
            
            return {
                'pii_detected': pii_found,
                'confidence_scores': confidence_scores,
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'total_pii_types': len(pii_found),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'text_length': len(text),
                'methods_used': list(set([pii['method'] for pii in pii_found.values()]))
            }
            
        except Exception as e:
            logger.error(f"Error in PII extraction: {e}")
            return {
                'error': f"PII extraction failed: {str(e)}",
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def _extract_spacy_entities(self, text: str) -> Optional[Dict]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return None
        
        try:
            doc = self.nlp(text)
            entities = {}
            
            for ent in doc.ents:
                entity_type = ent.label_.lower()
                entity_text = ent.text.strip()
                
                # Map spaCy labels to PII types
                if entity_type in ['person', 'per']:
                    pii_type = 'person_name'
                elif entity_type in ['org']:
                    pii_type = 'organization'
                elif entity_type in ['gpe']:
                    pii_type = 'location'
                elif entity_type in ['money']:
                    pii_type = 'financial_amount'
                else:
                    pii_type = entity_type
                
                if pii_type not in entities:
                    entities[pii_type] = {
                        'matches': [],
                        'count': 0,
                        'confidence': 0
                    }
                
                entities[pii_type]['matches'].append(entity_text)
                entities[pii_type]['count'] += 1
                entities[pii_type]['confidence'] = max(
                    entities[pii_type]['confidence'], 
                    float(getattr(ent, 'confidence', 0.7))
                )
            
            return entities
            
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
            return None
    
    def _extract_huggingface_entities(self, text: str) -> Optional[Dict]:
        """Extract entities using HuggingFace NER pipeline"""
        if not self.ner_pipeline:
            return None
        
        try:
            # Split text into chunks to handle long texts
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            all_entities = {}
            
            for chunk in chunks:
                entities = self.ner_pipeline(chunk)
                
                for entity in entities:
                    entity_type = entity['entity_group'].lower()
                    entity_text = entity['word'].strip()
                    confidence = entity['score']
                    
                    # Map BERT NER labels to PII types
                    if entity_type in ['per', 'person']:
                        pii_type = 'person_name'
                    elif entity_type in ['org']:
                        pii_type = 'organization'
                    elif entity_type in ['loc']:
                        pii_type = 'location'
                    elif entity_type in ['misc']:
                        pii_type = 'miscellaneous'
                    else:
                        pii_type = entity_type
                    
                    if pii_type not in all_entities:
                        all_entities[pii_type] = {
                            'matches': [],
                            'count': 0,
                            'confidence': 0
                        }
                    
                    all_entities[pii_type]['matches'].append(entity_text)
                    all_entities[pii_type]['count'] += 1
                    all_entities[pii_type]['confidence'] = max(
                        all_entities[pii_type]['confidence'],
                        confidence
                    )
            
            return all_entities
            
        except Exception as e:
            logger.error(f"HuggingFace entity extraction failed: {e}")
            return None
    
    def analyze_blockchain_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for blockchain-related content and sentiment
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing blockchain content analysis
        """
        try:
            results = {
                'sentiment_analysis': self._analyze_sentiment(text),
                'blockchain_keywords': self._extract_blockchain_keywords(text),
                'risk_indicators': self._detect_risk_indicators(text),
                'content_classification': self._classify_content(text),
                'text_statistics': self._calculate_text_statistics(text),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            # Calculate overall blockchain relevance score
            relevance_score = self._calculate_blockchain_relevance(results)
            results['blockchain_relevance_score'] = relevance_score
            results['blockchain_relevance_level'] = self._get_relevance_level(relevance_score)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in blockchain content analysis: {e}")
            return {
                'error': f"Content analysis failed: {str(e)}",
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple methods"""
        sentiment_results = {}
        
        # NLTK VADER sentiment
        if self.sentiment_analyzer:
            try:
                vader_scores = self.sentiment_analyzer.polarity_scores(text)
                sentiment_results['vader'] = {
                    'compound': vader_scores['compound'],
                    'positive': vader_scores['pos'],
                    'negative': vader_scores['neg'],
                    'neutral': vader_scores['neu']
                }
            except Exception as e:
                logger.warning(f"VADER sentiment analysis failed: {e}")
        
        # HuggingFace sentiment
        if self.classification_pipeline:
            try:
                hf_sentiment = self.classification_pipeline(text)
                sentiment_results['roberta'] = {
                    'label': hf_sentiment[0]['label'],
                    'score': hf_sentiment[0]['score']
                }
            except Exception as e:
                logger.warning(f"RoBERTa sentiment analysis failed: {e}")
        
        return sentiment_results
    
    def _extract_blockchain_keywords(self, text: str) -> Dict[str, Any]:
        """Extract blockchain-related keywords and phrases"""
        text_lower = text.lower()
        found_keywords = {}
        
        for category, keywords in self.blockchain_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches.append(keyword)
            
            if matches:
                found_keywords[category] = {
                    'matches': matches,
                    'count': len(matches)
                }
        
        return found_keywords
    
    def _detect_risk_indicators(self, text: str) -> Dict[str, Any]:
        """Detect potential risk indicators in text"""
        text_lower = text.lower()
        risk_indicators = {}
        
        # Urgency indicators
        urgency_terms = ['urgent', 'immediately', 'now', 'quickly', 'asap', 'emergency']
        urgency_matches = [term for term in urgency_terms if term in text_lower]
        if urgency_matches:
            risk_indicators['urgency'] = {
                'matches': urgency_matches,
                'risk_level': 'medium'
            }
        
        # Financial pressure
        pressure_terms = ['limited time', 'exclusive', 'guaranteed', 'double your money', 'get rich']
        pressure_matches = [term for term in pressure_terms if term in text_lower]
        if pressure_matches:
            risk_indicators['financial_pressure'] = {
                'matches': pressure_matches,
                'risk_level': 'high'
            }
        
        # Request for credentials
        credential_terms = ['private key', 'seed phrase', 'password', 'login', 'credentials']
        credential_matches = [term for term in credential_terms if term in text_lower]
        if credential_matches:
            risk_indicators['credential_request'] = {
                'matches': credential_matches,
                'risk_level': 'critical'
            }
        
        return risk_indicators
    
    def _classify_content(self, text: str) -> Dict[str, Any]:
        """Classify content type and purpose"""
        classification = {
            'content_type': 'unknown',
            'purpose': 'unknown',
            'confidence': 0.0
        }
        
        text_lower = text.lower()
        
        # Content type classification
        if any(term in text_lower for term in ['transaction', 'transfer', 'send', 'receive']):
            classification['content_type'] = 'transaction_related'
            classification['confidence'] += 0.3
        
        if any(term in text_lower for term in ['invest', 'trading', 'buy', 'sell']):
            classification['content_type'] = 'investment_related'
            classification['confidence'] += 0.3
        
        if any(term in text_lower for term in ['support', 'help', 'issue', 'problem']):
            classification['content_type'] = 'support_related'
            classification['confidence'] += 0.2
        
        # Purpose classification
        if any(term in text_lower for term in ['phishing', 'scam', 'fake', 'fraud']):
            classification['purpose'] = 'potentially_malicious'
            classification['confidence'] += 0.4
        
        if any(term in text_lower for term in ['verify', 'confirm', 'update', 'security']):
            classification['purpose'] = 'verification_request'
            classification['confidence'] += 0.3
        
        return classification
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        }
    
    def _calculate_pii_risk_score(self, pii_found: Dict, confidence_scores: Dict) -> float:
        """Calculate overall PII risk score"""
        if not pii_found:
            return 0.0
        
        risk_weights = {
            'ssn': 1.0,
            'credit_card': 1.0,
            'email': 0.6,
            'phone': 0.7,
            'person_name': 0.5,
            'bitcoin_address': 0.8,
            'ethereum_address': 0.8,
            'ip_address': 0.4,
            'url': 0.3
        }
        
        total_risk = 0.0
        for pii_type, data in pii_found.items():
            weight = risk_weights.get(pii_type, 0.5)
            confidence = confidence_scores.get(pii_type, 0.7)
            count_factor = min(data['count'] / 5, 1.0)  # Normalize count
            
            total_risk += weight * confidence * count_factor
        
        return min(total_risk, 1.0)  # Cap at 1.0
    
    def _calculate_blockchain_relevance(self, analysis_results: Dict) -> float:
        """Calculate blockchain relevance score"""
        relevance_score = 0.0
        
        # Keywords contribute to relevance
        blockchain_keywords = analysis_results.get('blockchain_keywords', {})
        for category, data in blockchain_keywords.items():
            category_weights = {
                'wallet_terms': 0.3,
                'transaction_terms': 0.3,
                'exchange_terms': 0.2,
                'defi_terms': 0.2,
                'suspicious_terms': 0.1,
                'scam_terms': 0.1
            }
            weight = category_weights.get(category, 0.1)
            relevance_score += weight * min(data['count'] / 3, 1.0)
        
        # Risk indicators contribute negatively to legitimacy
        risk_indicators = analysis_results.get('risk_indicators', {})
        if risk_indicators:
            relevance_score += 0.1  # Slight increase for being blockchain-related but risky
        
        return min(relevance_score, 1.0)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _get_relevance_level(self, relevance_score: float) -> str:
        """Convert relevance score to relevance level"""
        if relevance_score >= 0.7:
            return 'highly_relevant'
        elif relevance_score >= 0.4:
            return 'relevant'
        elif relevance_score >= 0.2:
            return 'somewhat_relevant'
        else:
            return 'not_relevant'
    
    def analyze_transaction_memo(self, memo: str) -> Dict[str, Any]:
        """
        Specialized analysis for transaction memos/notes
        
        Args:
            memo: Transaction memo or note text
            
        Returns:
            Dictionary containing memo analysis results
        """
        try:
            # Basic analysis
            pii_results = self.extract_pii_from_text(memo)
            content_results = self.analyze_blockchain_content(memo)
            
            # Memo-specific analysis
            memo_analysis = {
                'is_suspicious': False,
                'suspicion_reasons': [],
                'memo_type': 'standard',
                'contains_instructions': False,
                'urgency_level': 'normal'
            }
            
            memo_lower = memo.lower()
            
            # Check for suspicious patterns
            if any(term in memo_lower for term in ['mixer', 'privacy', 'anonymous', 'untraceable']):
                memo_analysis['is_suspicious'] = True
                memo_analysis['suspicion_reasons'].append('privacy_service_reference')
            
            if pii_results['risk_level'] in ['high', 'critical']:
                memo_analysis['is_suspicious'] = True
                memo_analysis['suspicion_reasons'].append('high_pii_risk')
            
            # Determine memo type
            if any(term in memo_lower for term in ['payment', 'invoice', 'bill']):
                memo_analysis['memo_type'] = 'payment'
            elif any(term in memo_lower for term in ['refund', 'return']):
                memo_analysis['memo_type'] = 'refund'
            elif any(term in memo_lower for term in ['test', 'testing']):
                memo_analysis['memo_type'] = 'test'
            
            # Check for instructions
            if any(term in memo_lower for term in ['send to', 'forward to', 'transfer to']):
                memo_analysis['contains_instructions'] = True
            
            # Assess urgency
            risk_indicators = content_results.get('risk_indicators', {})
            if 'urgency' in risk_indicators:
                memo_analysis['urgency_level'] = 'high'
            
            return {
                'memo_text': memo,
                'memo_analysis': memo_analysis,
                'pii_results': pii_results,
                'content_analysis': content_results,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in memo analysis: {e}")
            return {
                'error': f"Memo analysis failed: {str(e)}",
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def batch_analyze_text(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple texts in batch for efficiency
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary containing batch analysis results
        """
        try:
            batch_results = []
            summary_stats = {
                'total_texts': len(texts),
                'pii_detected_count': 0,
                'high_risk_count': 0,
                'blockchain_relevant_count': 0,
                'processing_errors': 0
            }
            
            for i, text in enumerate(texts):
                try:
                    # Analyze each text
                    pii_results = self.extract_pii_from_text(text)
                    content_results = self.analyze_blockchain_content(text)
                    
                    result = {
                        'text_id': i,
                        'text_preview': text[:100] + '...' if len(text) > 100 else text,
                        'pii_results': pii_results,
                        'content_analysis': content_results
                    }
                    
                    batch_results.append(result)
                    
                    # Update summary stats
                    if pii_results.get('total_pii_types', 0) > 0:
                        summary_stats['pii_detected_count'] += 1
                    
                    if pii_results.get('risk_level') in ['high', 'critical']:
                        summary_stats['high_risk_count'] += 1
                    
                    if content_results.get('blockchain_relevance_level') in ['relevant', 'highly_relevant']:
                        summary_stats['blockchain_relevant_count'] += 1
                        
                except Exception as e:
                    logger.error(f"Error analyzing text {i}: {e}")
                    summary_stats['processing_errors'] += 1
                    continue
            
            return {
                'batch_results': batch_results,
                'summary_statistics': summary_stats,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'models_used': self._get_available_models()
            }
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return {
                'error': f"Batch analysis failed: {str(e)}",
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def _get_available_models(self) -> List[str]:
        """Get list of available NLP models"""
        models = []
        if self.nlp:
            models.append('spacy')
        if self.sentiment_analyzer:
            models.append('nltk_vader')
        if self.ner_pipeline:
            models.append('huggingface_ner')
        if self.classification_pipeline:
            models.append('huggingface_classification')
        return models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models and capabilities"""
        return {
            'spacy_available': SPACY_AVAILABLE and self.nlp is not None,
            'nltk_available': NLTK_AVAILABLE and self.sentiment_analyzer is not None,
            'transformers_available': TRANSFORMERS_AVAILABLE and self.ner_pipeline is not None,
            'models_loaded': self._get_available_models(),
            'supported_languages': ['en'] if self.nlp else [],
            'pii_patterns_count': len(self.pii_patterns),
            'blockchain_keywords_count': sum(len(keywords) for keywords in self.blockchain_keywords.values()),
            'capabilities': [
                'pii_extraction',
                'sentiment_analysis',
                'blockchain_content_analysis',
                'transaction_memo_analysis',
                'batch_processing',
                'risk_assessment'
            ]
        }