"""
EVINCE: Evidence-Verified INtegrity Checker for ESG Claims

A novel framework for ESG-Washing detection in Vietnamese banking reports.

Key Features:
- Document-level Claim-Evidence Linking
- Deep Learning with Attention for explainability
- PhoBERT-based semantic understanding

Modules:
- models: ESGTopicClassifier, WashingDetector
- claim_evidence: DocumentAnalyzer, EvidenceRetriever, ClaimEvidenceMatcher
- training: Data loading and training pipeline
- evaluation: Metrics and evaluation utilities
"""

__version__ = "2.0.0"

from .models import (
    ESGTopicClassifier,
    ESGTopicClassifierInference,
    WashingDetector,
    WashingDetectorInference,
    ESG_LABELS,
    WASHING_LABELS
)

from .claim_evidence import (
    DocumentAnalyzer,
    EvidenceRetriever,
    ClaimEvidenceMatcher,
    SentenceClassifier,
    create_document_analyzer
)

__all__ = [
    # Version
    "__version__",
    # Models
    "ESGTopicClassifier",
    "ESGTopicClassifierInference",
    "WashingDetector",
    "WashingDetectorInference",
    "ESG_LABELS",
    "WASHING_LABELS",
    # Claim-Evidence
    "DocumentAnalyzer",
    "EvidenceRetriever",
    "ClaimEvidenceMatcher",
    "SentenceClassifier",
    "create_document_analyzer"
]
