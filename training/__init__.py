"""
EVINCE: Training Package

Contains:
- DataLoader: Load and preprocess data with LLM labeling
- Train: Training pipeline for models
"""

from .data_loader import (
    ESGDataset,
    ClaimEvidenceDataset,
    create_esg_dataloader,
    load_sentences_from_csv,
    filter_noise_sentences,
    create_train_val_split
)

from .train import (
    Trainer,
    train_esg_classifier,
    train_washing_detector
)

__all__ = [
    # Data Loading
    "ESGDataset",
    "ClaimEvidenceDataset",
    "create_esg_dataloader",
    "load_sentences_from_csv",
    "filter_noise_sentences",
    "create_train_val_split",
    # Training
    "Trainer",
    "train_esg_classifier",
    "train_washing_detector"
]
