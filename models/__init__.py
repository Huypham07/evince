"""
EVINCE Models Package

Contains:
- ESGTopicClassifier: Multi-class ESG topic classification
- WashingDetector: ESG-washing detection with attention
"""

from .esg_topic_classifier import (
    ESGTopicClassifier,
    ESGTopicClassifierInference,
    HuggingFaceESGClassifierInference,
    ESGClassificationResult,
    ESG_LABELS,
    ESG_LABELS_VN,
    LABEL_TO_ID,
    ID_TO_LABEL,
    HUGGINGFACE_ESG_MODEL,
    create_esg_classifier,
    load_esg_classifier_from_huggingface
)

from .washing_detector import (
    WashingDetector,
    WashingDetectorInference,
    WashingDetectionResult,
    WASHING_LABELS,
    WASHING_LABELS_VN,
    WASHING_LABEL_TO_ID,
    WASHING_ID_TO_LABEL,
    create_washing_detector
)

__all__ = [
    # ESG Classifier
    "ESGTopicClassifier",
    "ESGTopicClassifierInference", 
    "HuggingFaceESGClassifierInference",
    "ESGClassificationResult",
    "ESG_LABELS",
    "ESG_LABELS_VN",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "HUGGINGFACE_ESG_MODEL",
    "create_esg_classifier",
    "load_esg_classifier_from_huggingface",
    # Washing Detector
    "WashingDetector",
    "WashingDetectorInference",
    "WashingDetectionResult",
    "WASHING_LABELS",
    "WASHING_LABELS_VN",
    "WASHING_LABEL_TO_ID",
    "WASHING_ID_TO_LABEL",
    "create_washing_detector"
]
