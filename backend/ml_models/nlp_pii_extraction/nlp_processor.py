"""
NLP and PII Extraction Module for ETHEREYE
==========================================

This module implements Natural Language Processing and Personally Identifiable Information (PII) 
extraction for blockchain analytics using spaCy, NLTK, and HuggingFace Transformers.

Features:
- PII detection from transaction metadata
- Sentiment analysis of social media posts
- Named Entity Recognition (NER)
- Text classification for suspicious content
- Social media analysis for wallet associations
- Email/phone/address extraction
"""

import re
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# NLP Libraries
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# HuggingFace Transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModelForTokenClassification
)
import torch

# Text processing utilities
from textblob import TextBlob
import hashlib

logger = logging.getLogger(__name__)

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

@dataclass
class PIIResult:
    """Container for PII extraction results"""
    text: str
    pii_entities: List[Dict[str, Any]]
    pii_types_found: List[str]
    pii_confidence_scores: Dict[str, float]
    sanitized_text: str
    risk_score: float

@dataclass
class TextAnalysisResult:
    """Container for comprehensive text analysis results"""
    original_text: str
    cleaned_text: str
    sentiment: Dict[str, float]
    entities: List[Dict[str, Any]]
    pii_data: PIIResult
    language: str
    toxicity_score: float
    keywords: List[str]
    classification_results: Dict[str, Any]

class NLPPIIExtractor:
    """
    Comprehensive NLP and PII extraction for blockchain intelligence
    """
    
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 use_gpu: bool = False):
        """
        Initialize the NLP processor
        
        Args:
            spacy_model: spaCy model to use for NER
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu
        self.spacy_model_name = spacy_model
        
        # Initialize models
        self._initialize_models()
        
        # PII patterns
        self._compile_pii_patterns()
        
        # Keywords for blockchain/crypto context
        self.crypto_keywords = [
            'bitcoin', 'ethereum', 'crypto', 'blockchain', 'wallet', 'address',
            'transaction', 'mining', 'defi', 'nft', 'exchange', 'trading',
            'laundering', 'suspicious', 'fraud', 'scam', 'hack'
        ]
    
    def _initialize_models(self):
        """Initialize all NLP models"""
        logger.info("Initializing NLP models...")
        
        try:
            # Load spaCy model
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
        except IOError:
            logger.warning(f"spaCy model {self.spacy_model_name} not found. Please install it.")
            self.nlp = None
        
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        # Initialize NLTK components
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize HuggingFace models
        self._initialize_hf_models()
        
        logger.info("NLP models initialized successfully")
    
    def _initialize_hf_models(self):
        """Initialize HuggingFace Transformers models"""
        device = 0 if self.use_gpu and torch.cuda.is_available() else -1
        
        try:
            # Sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device
            )
            
            # Toxicity detection pipeline
            self.toxicity_pipeline = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=device
            )
            
            # NER pipeline for PII detection
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=device
            )
            
            logger.info("HuggingFace models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Error loading HuggingFace models: {e}")
            self.sentiment_pipeline = None
            self.toxicity_pipeline = None
            self.ner_pipeline = None
    
    def _compile_pii_patterns(self):
        """Compile regex patterns for PII detection"""
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'ethereum_address': re.compile(r'\b0x[a-fA-F0-9]{40}\b'),
            'bitcoin_address': re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'iban': re.compile(r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}[A-Z0-9]{1,16}\b'),
            'routing_number': re.compile(r'\b[0-9]{9}\b')
        }
    
    def extract_pii(self, text: str) -> PIIResult:
        """
        Extract PII from text using multiple techniques
        
        Args:
            text: Input text to analyze
            
        Returns:
            PIIResult object with extracted PII information
        """
        pii_entities = []
        pii_types_found = []
        confidence_scores = {}
        
        # Regex-based PII extraction
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                entity = {
                    'type': pii_type,
                    'value': match,
                    'method': 'regex',
                    'confidence': 0.9,
                    'start': text.find(match),
                    'end': text.find(match) + len(match)
                }
                pii_entities.append(entity)
                
                if pii_type not in pii_types_found:
                    pii_types_found.append(pii_type)
                    confidence_scores[pii_type] = 0.9
        
        # spaCy NER for additional entity detection
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']:
                    entity = {
                        'type': ent.label_.lower(),
                        'value': ent.text,
                        'method': 'spacy_ner',
                        'confidence': 0.8,
                        'start': ent.start_char,
                        'end': ent.end_char
                    }
                    pii_entities.append(entity)
                    
                    if ent.label_.lower() not in pii_types_found:
                        pii_types_found.append(ent.label_.lower())
                        confidence_scores[ent.label_.lower()] = 0.8
        
        # HuggingFace NER for additional detection
        if self.ner_pipeline:
            try:
                hf_entities = self.ner_pipeline(text)
                for ent in hf_entities:
                    entity = {
                        'type': ent['entity_group'].lower(),
                        'value': ent['word'],
                        'method': 'huggingface_ner',
                        'confidence': ent['score'],
                        'start': ent['start'],
                        'end': ent['end']
                    }
                    pii_entities.append(entity)
                    
                    if ent['entity_group'].lower() not in pii_types_found:
                        pii_types_found.append(ent['entity_group'].lower())
                        confidence_scores[ent['entity_group'].lower()] = ent['score']
            except Exception as e:
                logger.warning(f"Error in HuggingFace NER: {e}")
        
        # Create sanitized text
        sanitized_text = self._sanitize_text(text, pii_entities)
        
        # Calculate risk score based on PII types found
        risk_score = self._calculate_pii_risk_score(pii_types_found, confidence_scores)
        
        return PIIResult(
            text=text,
            pii_entities=pii_entities,
            pii_types_found=pii_types_found,
            pii_confidence_scores=confidence_scores,
            sanitized_text=sanitized_text,
            risk_score=risk_score
        )
    
    def _sanitize_text(self, text: str, pii_entities: List[Dict]) -> str:
        """
        Sanitize text by replacing PII with placeholders
        """
        sanitized = text
        
        # Sort entities by start position in reverse order to avoid index shifting
        sorted_entities = sorted(pii_entities, key=lambda x: x['start'], reverse=True)
        
        for entity in sorted_entities:
            placeholder = f"[{entity['type'].upper()}]"
            start, end = entity['start'], entity['end']
            sanitized = sanitized[:start] + placeholder + sanitized[end:]
        
        return sanitized
    
    def _calculate_pii_risk_score(self, pii_types: List[str], 
                                  confidence_scores: Dict[str, float]) -> float:
        """
        Calculate risk score based on types of PII found
        """
        high_risk_types = ['ssn', 'credit_card', 'email', 'phone']
        medium_risk_types = ['ethereum_address', 'bitcoin_address', 'iban']
        low_risk_types = ['person', 'org', 'date']
        
        risk_score = 0.0
        
        for pii_type in pii_types:
            confidence = confidence_scores.get(pii_type, 0.5)
            
            if pii_type in high_risk_types:
                risk_score += 0.3 * confidence
            elif pii_type in medium_risk_types:
                risk_score += 0.2 * confidence
            elif pii_type in low_risk_types:
                risk_score += 0.1 * confidence
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using multiple approaches
        """
        sentiment_results = {}
        
        # NLTK VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        sentiment_results['vader'] = vader_scores
        
        # TextBlob sentiment
        blob = TextBlob(text)
        sentiment_results['textblob'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # HuggingFace sentiment
        if self.sentiment_pipeline:
            try:
                hf_sentiment = self.sentiment_pipeline(text)
                sentiment_results['huggingface'] = hf_sentiment[0]
            except Exception as e:
                logger.warning(f"Error in HuggingFace sentiment analysis: {e}")
        
        # Calculate combined sentiment score
        compound_score = (
            vader_scores['compound'] * 0.4 +
            blob.sentiment.polarity * 0.4 +
            (sentiment_results.get('huggingface', {}).get('score', 0) * 
             (1 if sentiment_results.get('huggingface', {}).get('label') == 'POSITIVE' else -1) * 0.2)
        )
        
        sentiment_results['combined_score'] = compound_score
        sentiment_results['combined_label'] = 'positive' if compound_score > 0.05 else 'negative' if compound_score < -0.05 else 'neutral'
        
        return sentiment_results
    
    def detect_toxicity(self, text: str) -> Dict[str, float]:
        """
        Detect toxic/harmful content
        """
        toxicity_result = {'score': 0.0, 'is_toxic': False}
        
        if self.toxicity_pipeline:
            try:
                result = self.toxicity_pipeline(text)
                toxicity_result['score'] = result[0]['score'] if result[0]['label'] == 'TOXIC' else 1 - result[0]['score']
                toxicity_result['is_toxic'] = result[0]['label'] == 'TOXIC'
                toxicity_result['confidence'] = result[0]['score']
            except Exception as e:
                logger.warning(f"Error in toxicity detection: {e}")
        
        return toxicity_result
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract keywords from text
        """
        # Clean and tokenize text
        words = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        keywords = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word.isalpha() and word not in self.stop_words and len(word) > 2
        ]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(keywords)
        
        # Get top keywords
        top_keywords = [word for word, count in word_freq.most_common(top_k)]
        
        # Add crypto-related keywords if found
        crypto_found = [kw for kw in self.crypto_keywords if kw in text.lower()]
        
        # Combine and deduplicate
        all_keywords = list(set(top_keywords + crypto_found))
        
        return all_keywords[:top_k]
    
    def classify_text_content(self, text: str) -> Dict[str, Any]:
        """
        Classify text content for blockchain/crypto relevance
        """
        classification_result = {
            'is_crypto_related': False,
            'crypto_relevance_score': 0.0,
            'detected_categories': [],
            'risk_indicators': []
        }
        
        text_lower = text.lower()
        
        # Check for crypto keywords
        crypto_matches = [kw for kw in self.crypto_keywords if kw in text_lower]
        classification_result['is_crypto_related'] = len(crypto_matches) > 0
        classification_result['crypto_relevance_score'] = min(len(crypto_matches) / 5, 1.0)
        
        # Categorize content
        categories = {
            'trading': ['buy', 'sell', 'trade', 'exchange', 'price', 'market'],
            'technical': ['smart contract', 'mining', 'consensus', 'blockchain', 'node'],
            'security': ['hack', 'breach', 'stolen', 'fraud', 'scam', 'phishing'],
            'regulatory': ['regulation', 'compliance', 'legal', 'government', 'ban'],
            'suspicious': ['money laundering', 'illicit', 'dark web', 'mixer', 'tumbler']
        }
        
        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                classification_result['detected_categories'].append(category)
        
        # Identify risk indicators
        risk_keywords = ['suspicious', 'fraud', 'scam', 'hack', 'stolen', 'illicit', 'laundering']
        classification_result['risk_indicators'] = [kw for kw in risk_keywords if kw in text_lower]
        
        return classification_result
    
    def analyze_text(self, text: str) -> TextAnalysisResult:
        """
        Comprehensive text analysis combining all features
        
        Args:
            text: Input text to analyze
            
        Returns:
            TextAnalysisResult with comprehensive analysis
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract PII
        pii_result = self.extract_pii(text)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(cleaned_text)
        
        # Extract entities
        entities = []
        if self.nlp:
            doc = self.nlp(cleaned_text)
            entities = [
                {
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                for ent in doc.ents
            ]
        
        # Detect language
        language = 'en'  # Default
        if self.nlp:
            language = self.nlp.lang
        
        # Detect toxicity
        toxicity = self.detect_toxicity(cleaned_text)
        
        # Extract keywords
        keywords = self.extract_keywords(cleaned_text)
        
        # Classify content
        classification = self.classify_text_content(cleaned_text)
        
        return TextAnalysisResult(
            original_text=text,
            cleaned_text=cleaned_text,
            sentiment=sentiment,
            entities=entities,
            pii_data=pii_result,
            language=language,
            toxicity_score=toxicity['score'],
            keywords=keywords,
            classification_results=classification
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        """
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (but keep for PII detection)
        # cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[!?]{2,}', '!', cleaned)
        
        # Strip and return
        return cleaned.strip()
    
    def batch_analyze_texts(self, texts: List[str]) -> List[TextAnalysisResult]:
        """
        Analyze multiple texts in batch
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of TextAnalysisResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_text(text)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                # Add empty result to maintain list length
                results.append(None)
        
        logger.info(f"Batch analysis completed. Processed {len(texts)} texts")
        return results
    
    def extract_social_media_insights(self, texts: List[str], 
                                     metadata: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Extract insights from social media texts related to crypto/blockchain
        
        Args:
            texts: List of social media posts/comments
            metadata: Optional metadata for each text (timestamps, user info, etc.)
            
        Returns:
            Dictionary with social media insights
        """
        insights = {
            'total_posts': len(texts),
            'crypto_related_posts': 0,
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'top_keywords': [],
            'risk_posts': [],
            'pii_found': 0,
            'entities_found': {},
            'temporal_analysis': {},
            'user_insights': {}
        }
        
        # Analyze all texts
        results = self.batch_analyze_texts(texts)
        
        # Process results
        all_keywords = []
        entity_counts = {}
        
        for i, result in enumerate(results):
            if result is None:
                continue
            
            # Count crypto-related posts
            if result.classification_results['is_crypto_related']:
                insights['crypto_related_posts'] += 1
            
            # Sentiment distribution
            sentiment_label = result.sentiment['combined_label']
            insights['sentiment_distribution'][sentiment_label] += 1
            
            # Collect keywords
            all_keywords.extend(result.keywords)
            
            # Count entities
            for entity in result.entities:
                label = entity['label']
                entity_counts[label] = entity_counts.get(label, 0) + 1
            
            # Risk posts
            if (result.pii_data.risk_score > 0.3 or 
                result.toxicity_score > 0.7 or 
                len(result.classification_results['risk_indicators']) > 0):
                
                risk_post = {
                    'index': i,
                    'text': result.original_text[:200] + '...' if len(result.original_text) > 200 else result.original_text,
                    'risk_score': result.pii_data.risk_score,
                    'toxicity_score': result.toxicity_score,
                    'risk_indicators': result.classification_results['risk_indicators'],
                    'pii_types': result.pii_data.pii_types_found
                }
                insights['risk_posts'].append(risk_post)
            
            # PII count
            if result.pii_data.pii_entities:
                insights['pii_found'] += 1
        
        # Calculate top keywords
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        insights['top_keywords'] = [
            {'keyword': kw, 'count': count} 
            for kw, count in keyword_counts.most_common(20)
        ]
        
        # Entity insights
        insights['entities_found'] = entity_counts
        
        # Temporal analysis if metadata provided
        if metadata:
            insights['temporal_analysis'] = self._analyze_temporal_patterns(results, metadata)
        
        return insights
    
    def _analyze_temporal_patterns(self, results: List[TextAnalysisResult], 
                                   metadata: List[Dict]) -> Dict[str, Any]:
        """
        Analyze temporal patterns in social media data
        """
        temporal_insights = {
            'posting_frequency': {},
            'sentiment_trends': {},
            'keyword_trends': {}
        }
        
        # This would require timestamp data in metadata
        # Implementation depends on metadata structure
        
        return temporal_insights
    
    def generate_pii_report(self, text: str) -> Dict[str, Any]:
        """
        Generate a comprehensive PII report for a given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Comprehensive PII report
        """
        pii_result = self.extract_pii(text)
        
        report = {
            'text_length': len(text),
            'analysis_timestamp': datetime.now().isoformat(),
            'pii_summary': {
                'total_entities_found': len(pii_result.pii_entities),
                'unique_pii_types': len(pii_result.pii_types_found),
                'risk_score': pii_result.risk_score,
                'risk_level': self._get_risk_level(pii_result.risk_score)
            },
            'detailed_findings': [],
            'recommendations': [],
            'sanitized_text': pii_result.sanitized_text,
            'text_hash': hashlib.sha256(text.encode()).hexdigest()[:16]  # Shortened hash for identification
        }
        
        # Detailed findings
        for entity in pii_result.pii_entities:
            finding = {
                'type': entity['type'],
                'value': entity['value'][:10] + '***' if len(entity['value']) > 10 else '***',  # Partial masking
                'confidence': entity['confidence'],
                'method': entity['method'],
                'position': f"{entity['start']}-{entity['end']}"
            }
            report['detailed_findings'].append(finding)
        
        # Recommendations
        if pii_result.risk_score > 0.7:
            report['recommendations'].append("HIGH RISK: Multiple sensitive PII types detected. Consider full sanitization.")
        elif pii_result.risk_score > 0.4:
            report['recommendations'].append("MEDIUM RISK: Some PII detected. Review and sanitize as needed.")
        elif pii_result.risk_score > 0.1:
            report['recommendations'].append("LOW RISK: Minimal PII detected. Monitor for context.")
        else:
            report['recommendations'].append("No significant PII detected.")
        
        if 'email' in pii_result.pii_types_found or 'phone' in pii_result.pii_types_found:
            report['recommendations'].append("Contact information found. Consider masking for privacy.")
        
        if 'ethereum_address' in pii_result.pii_types_found or 'bitcoin_address' in pii_result.pii_types_found:
            report['recommendations'].append("Cryptocurrency addresses found. Monitor for suspicious activity.")
        
        return report
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert numeric risk score to categorical level"""
        if risk_score > 0.7:
            return 'HIGH'
        elif risk_score > 0.4:
            return 'MEDIUM'
        elif risk_score > 0.1:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def export_analysis_results(self, results: List[TextAnalysisResult], 
                               output_path: str) -> None:
        """
        Export analysis results to CSV
        
        Args:
            results: List of analysis results
            output_path: Path to save CSV file
        """
        data = []
        
        for i, result in enumerate(results):
            if result is None:
                continue
                
            row = {
                'text_id': i,
                'original_text_length': len(result.original_text),
                'cleaned_text_length': len(result.cleaned_text),
                'sentiment_score': result.sentiment['combined_score'],
                'sentiment_label': result.sentiment['combined_label'],
                'toxicity_score': result.toxicity_score,
                'pii_entities_count': len(result.pii_data.pii_entities),
                'pii_risk_score': result.pii_data.risk_score,
                'crypto_related': result.classification_results['is_crypto_related'],
                'crypto_relevance_score': result.classification_results['crypto_relevance_score'],
                'entities_count': len(result.entities),
                'keywords': ', '.join(result.keywords[:5]),  # Top 5 keywords
                'risk_indicators': ', '.join(result.classification_results['risk_indicators']),
                'pii_types_found': ', '.join(result.pii_data.pii_types_found)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Analysis results exported to {output_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Model information dictionary
        """
        info = {
            'spacy_model': self.spacy_model_name,
            'spacy_loaded': self.nlp is not None,
            'use_gpu': self.use_gpu,
            'gpu_available': torch.cuda.is_available(),
            'huggingface_models': {
                'sentiment_pipeline': self.sentiment_pipeline is not None,
                'toxicity_pipeline': self.toxicity_pipeline is not None,
                'ner_pipeline': self.ner_pipeline is not None
            },
            'pii_patterns_count': len(self.pii_patterns),
            'crypto_keywords_count': len(self.crypto_keywords)
        }
        
        return info