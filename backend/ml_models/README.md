# ETHEREYE Machine Learning Pipeline

This directory contains a comprehensive machine learning pipeline for blockchain analytics and suspicious wallet classification. The implementation includes clustering, anomaly detection, NLP/PII extraction, and custom risk scoring algorithms.

## 🏗️ Architecture Overview

```
ml_models/
├── clustering/              # DBSCAN clustering for wallet grouping
├── anomaly_detection/       # Isolation Forest for suspicious activity
├── nlp_pii_extraction/     # NLP & PII analysis with spaCy/HuggingFace
├── risk_scoring/           # Custom ML pipeline for risk assessment
├── preprocessing/          # Data preprocessing and feature engineering
└── utils/                 # Model utilities and helpers
```

## 🔧 Implemented Algorithms

### 1. Clustering → DBSCAN
- **Purpose**: Group similar wallets and transaction patterns
- **Implementation**: `clustering/dbscan_model.py`
- **Features**:
  - Automatic parameter optimization
  - Silhouette score evaluation
  - Cluster visualization and statistics
  - Noise point detection for outliers

### 2. Anomaly Detection → IsolationForest
- **Purpose**: Detect suspicious transactions and wallet behaviors
- **Implementation**: `anomaly_detection/isolation_forest_model.py`
- **Features**:
  - Contamination parameter optimization
  - Cross-validation for model stability
  - Anomaly explanation and scoring
  - Performance evaluation metrics

### 3. NLP/PII Extraction → spaCy, NLTK, HuggingFace Transformers
- **Purpose**: Extract PII and analyze text for suspicious content
- **Implementation**: `nlp_pii_extraction/nlp_processor.py`
- **Features**:
  - Multi-model PII detection (regex, spaCy, HuggingFace)
  - Sentiment analysis with multiple approaches
  - Toxicity detection using BERT models
  - Cryptocurrency-specific text classification
  - Social media analysis and insights

### 4. Risk Scoring → Custom ML Pipeline
- **Purpose**: Comprehensive risk assessment combining all models
- **Implementation**: `risk_scoring/risk_model.py`
- **Features**:
  - Multi-factor risk assessment
  - Configurable feature weights
  - Ensemble scoring methodology
  - Actionable recommendations
  - Batch processing capabilities

## 📊 Core Components

### Data Preprocessing (`preprocessing/data_processor.py`)
- Transaction data preprocessing
- Address-based feature engineering
- Temporal pattern extraction
- Behavioral metrics calculation
- Missing value handling and normalization

### Model Utilities (`utils/model_utils.py`)
- Model persistence and loading
- Performance evaluation metrics
- Data validation and schema checking
- Training report generation
- Data drift detection

## 🚀 API Endpoints

All ML functionality is exposed through FastAPI endpoints at `/api/v1/ml/`:

### Clustering
- `POST /clustering/analyze` - Perform DBSCAN clustering
- Parameters: data, eps, min_samples, feature_columns

### Anomaly Detection
- `POST /anomaly-detection/analyze` - Detect anomalies
- Parameters: data, contamination, n_estimators, feature_columns

### NLP Analysis
- `POST /nlp/analyze` - Comprehensive text analysis
- Parameters: texts, extract_pii, analyze_sentiment, detect_toxicity

### Risk Assessment
- `POST /risk-assessment/analyze` - Single wallet assessment
- `POST /risk-assessment/batch` - Batch wallet assessment
- Parameters: wallet_address, transaction_data, text_data, config

### Data Preprocessing
- `POST /preprocessing/validate` - Validate data schema
- `POST /preprocessing/clean` - Clean and preprocess data

### Utilities
- `GET /models/info` - Get model information
- `POST /models/initialize` - Initialize ML models
- `GET /statistics/risk-distribution` - Risk statistics

## 🔧 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Required Models
```bash
# spaCy model
python -m spacy download en_core_web_sm

# NLTK data (automatically downloaded on first use)
# HuggingFace models (automatically downloaded on first use)
```

### 3. Initialize Models
```python
from ml_models.risk_scoring.risk_model import RiskScorer

# Initialize the complete pipeline
risk_scorer = RiskScorer()
risk_scorer.initialize_models()
```

## 📈 Training & Evaluation

### Training Script
Use the comprehensive training script to train all models:

```bash
python scripts/train_models.py --data-path /path/to/data --output-dir trained_models
```

### Training Data Format
Expected data files in the training directory:
- `transactions.csv` - Transaction data
- `addresses.csv` - Address data  
- `text_data.json` - Text samples for NLP
- `risk_labels.csv` - Optional risk labels

### Model Evaluation
The training script automatically:
- Optimizes hyperparameters
- Evaluates model performance
- Generates comprehensive reports
- Saves trained models with version control

## 🎯 Usage Examples

### 1. Clustering Analysis
```python
from ml_models.clustering.dbscan_model import DBSCANClustering
import pandas as pd

# Load data
data = pd.read_csv('wallet_data.csv')

# Initialize and fit model
clustering = DBSCANClustering(eps=0.5, min_samples=5)
results = clustering.fit(data)

print(f"Found {results.n_clusters} clusters")
print(f"Noise points: {results.n_noise}")
```

### 2. Anomaly Detection
```python
from ml_models.anomaly_detection.isolation_forest_model import IsolationForestDetector

# Initialize detector
detector = IsolationForestDetector(contamination=0.1)

# Fit and predict
results = detector.fit_predict(transaction_data)
anomalies = results.anomaly_indices

print(f"Detected {len(anomalies)} anomalous transactions")
```

### 3. NLP Analysis
```python
from ml_models.nlp_pii_extraction.nlp_processor import NLPPIIExtractor

# Initialize processor
nlp_processor = NLPPIIExtractor()

# Analyze texts
texts = ["Sample social media post with wallet info..."]
results = nlp_processor.batch_analyze_texts(texts)

# Extract insights
insights = nlp_processor.extract_social_media_insights(texts)
```

### 4. Risk Assessment
```python
from ml_models.risk_scoring.risk_model import RiskScorer

# Initialize risk scorer
risk_scorer = RiskScorer()
risk_scorer.initialize_models()

# Assess wallet risk
assessment = risk_scorer.assess_wallet_risk(
    wallet_address="0x123...",
    wallet_data=transaction_df,
    text_data=social_media_posts
)

print(f"Risk Level: {assessment.risk_level}")
print(f"Risk Score: {assessment.overall_risk_score:.3f}")
```

## 📊 Model Performance

### DBSCAN Clustering
- **Metric**: Silhouette Score
- **Optimization**: Automated parameter tuning
- **Output**: Cluster assignments and statistics

### Isolation Forest
- **Metric**: Cross-validation stability
- **Optimization**: Contamination parameter tuning
- **Output**: Anomaly predictions and scores

### NLP Processing
- **PII Detection**: Multi-model ensemble approach
- **Sentiment Analysis**: Combined VADER, TextBlob, and HuggingFace
- **Toxicity Detection**: BERT-based classification

### Risk Scoring
- **Approach**: Weighted ensemble of all risk factors
- **Confidence**: Based on factor consistency and data availability
- **Output**: Risk level, score, and actionable recommendations

## 🔧 Configuration

### Risk Scoring Configuration
```python
config = {
    'high_risk_threshold': 0.7,
    'medium_risk_threshold': 0.4,
    'feature_weights': {
        'clustering': 0.15,
        'anomaly': 0.20,
        'nlp': 0.15,
        'transaction': 0.20,
        'behavioral': 0.15,
        'temporal': 0.10,
        'network': 0.05
    }
}

risk_scorer = RiskScorer(config=config)
```

### Model Initialization Options
```python
# GPU acceleration (if available)
nlp_processor = NLPPIIExtractor(use_gpu=True)

# Custom contamination rate
anomaly_detector = IsolationForestDetector(contamination=0.05)

# Custom clustering parameters
clustering_model = DBSCANClustering(eps=0.3, min_samples=10)
```

## 📝 Model Outputs

### Risk Assessment Output
```json
{
  "wallet_address": "0x123...",
  "overall_risk_score": 0.75,
  "risk_level": "HIGH",
  "confidence": 0.87,
  "risk_factors": {
    "clustering_risk": 0.6,
    "anomaly_risk": 0.8,
    "transaction_risk": 0.7,
    "behavioral_risk": 0.5,
    "temporal_risk": 0.4,
    "network_risk": 0.3,
    "compliance_risk": 0.9
  },
  "contributing_factors": [
    "Anomalous transaction patterns detected",
    "Potential compliance violations",
    "Suspicious transaction characteristics"
  ],
  "recommendations": [
    "IMMEDIATE ACTION REQUIRED: Flag for manual investigation",
    "File Suspicious Activity Report (SAR) if applicable",
    "Review against AML/KYC requirements"
  ]
}
```

## 🚨 Monitoring & Alerts

The pipeline includes built-in monitoring for:
- Model drift detection
- Performance degradation
- Data quality issues
- API endpoint health

## 🔒 Security & Privacy

- PII data is automatically sanitized
- All sensitive information is masked in logs
- Models can be trained on anonymized data
- GDPR-compliant PII handling

## 📚 Dependencies

### Core ML Libraries
- `scikit-learn` - Clustering and anomaly detection
- `spacy` - Named entity recognition
- `nltk` - Text preprocessing and sentiment analysis
- `transformers` - HuggingFace transformer models
- `torch` - PyTorch backend for transformers

### Additional Libraries
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `joblib` - Model serialization
- `scipy` - Statistical tests

## 🤝 Contributing

When adding new models or features:
1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Include example usage
5. Add API endpoints if needed

## 📄 License

This ML pipeline is part of the ETHEREYE project and follows the same licensing terms.

---

For more detailed information about specific components, refer to the individual module documentation and docstrings within each file.