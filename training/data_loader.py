"""
EVINCE: Data Loader

Data loading and preprocessing for ESG-washing detection.
Optimized for paragraph-level input with max_length=512.

Features:
- Load sentences/paragraphs from CSV
- LLM-based pseudo-labeling with Qwen3
- Create PyTorch datasets and dataloaders
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Maximum sequence length for paragraph-level input
MAX_SEQ_LENGTH = 512


@dataclass
class LabeledSentence:
    """A labeled sentence/paragraph for training."""
    sentence: str
    esg_label: Optional[str] = None  # E, S, G, Financing, Policy, Non-ESG
    sentence_type: Optional[str] = None  # CLAIM, EVIDENCE, CONTEXT, NON_ESG
    washing_type: Optional[str] = None  # 7 washing types
    bank: Optional[str] = None
    year: Optional[int] = None
    report_type: Optional[str] = None


def load_sentences_from_csv(
    csv_path: str,
    sample_size: Optional[int] = None,
    banks: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load sentences/paragraphs from CSV file with optional filtering.
    
    Args:
        csv_path: Path to CSV file
        sample_size: Number of sentences to sample (None = all)
        banks: List of banks to filter (None = all)
        years: List of years to filter (None = all)
        random_state: Random seed for sampling
        
    Returns:
        DataFrame with sentences
    """
    df = pd.read_csv(csv_path)
    
    # Filter by banks
    if banks:
        df['bank'] = df['bank'].astype(str).str.lower()
        banks_lower = [b.lower() for b in banks]
        df = df[df['bank'].isin(banks_lower)]
    
    # Filter by years
    if years:
        df = df[df['year'].isin(years)]
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    
    return df.reset_index(drop=True)


def filter_noise_sentences(df: pd.DataFrame, sentence_col: str = "sentence") -> pd.DataFrame:
    """
    Filter out noisy sentences (too short, numbers-only, headers, etc.).
    
    Args:
        df: DataFrame with sentences
        sentence_col: Column name for sentences
        
    Returns:
        Filtered DataFrame
    """
    def is_valid(text):
        if not isinstance(text, str):
            return False
        text = text.strip()
        # Too short
        if len(text) < 20:
            return False
        # Numbers only
        if text.replace(" ", "").replace(",", "").replace(".", "").isdigit():
            return False
        # Too few words
        if len(text.split()) < 5:
            return False
        return True
    
    mask = df[sentence_col].apply(is_valid)
    return df[mask].reset_index(drop=True)


class ESGDataset(Dataset):
    """
    PyTorch Dataset for ESG classification and washing detection.
    
    Optimized for paragraph-level input with max_length=512.
    
    Supports:
    - ESG topic classification (6 classes)
    - Sentence type classification (4 classes)
    - Washing type classification (7 classes)
    """
    
    def __init__(
        self,
        sentences: List[str],
        labels: List[int],
        tokenizer_name: str = "vinai/phobert-base-v2",
        max_length: int = MAX_SEQ_LENGTH,
        task: str = "esg_topic"  # "esg_topic", "sentence_type", "washing"
    ):
        """
        Initialize dataset.
        
        Args:
            sentences: List of sentence/paragraph texts
            labels: List of integer labels
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length (default 512 for paragraphs)
            task: Task type for logging
        """
        self.sentences = sentences
        self.labels = labels
        self.max_length = max_length
        self.task = task
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        # Handle None or invalid sentences
        if sentence is None or not isinstance(sentence, str):
            sentence = ""
        
        # Clean sentence - remove NaN, special chars that might cause issues
        sentence = str(sentence).strip()
        if not sentence:
            sentence = "[UNK]"
        
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Clamp token IDs to valid vocab range to prevent CUDA errors
        vocab_size = len(self.tokenizer)
        input_ids = torch.clamp(encoding["input_ids"].squeeze(0), min=0, max=vocab_size - 1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class ClaimEvidenceDataset(Dataset):
    """
    Dataset for claim-evidence pair matching.
    
    Input: (claim, evidence) pair
    Output: match_score (0 or 1)
    """
    
    def __init__(
        self,
        claims: List[str],
        evidences: List[str],
        labels: List[int],  # 1 = match, 0 = no match
        tokenizer_name: str = "vinai/phobert-base-v2",
        max_length: int = MAX_SEQ_LENGTH
    ):
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.max_length = max_length
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __len__(self):
        return len(self.claims)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        claim = self.claims[idx]
        evidence = self.evidences[idx]
        label = self.labels[idx]
        
        # Pair encoding: [CLS] claim [SEP] evidence [SEP]
        encoding = self.tokenizer(
            claim,
            evidence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }


def create_esg_dataloader(
    sentences: List[str],
    labels: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_length: int = MAX_SEQ_LENGTH,
    task: str = "esg_topic"
) -> DataLoader:
    """
    Create DataLoader for ESG tasks.
    
    Args:
        sentences: List of sentence/paragraph texts
        labels: List of integer labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        max_length: Maximum sequence length (default 512 for paragraphs)
        task: Task type
        
    Returns:
        PyTorch DataLoader
    """
    dataset = ESGDataset(
        sentences=sentences,
        labels=labels,
        max_length=max_length,
        task=task
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def create_train_val_split(
    sentences: List[str],
    labels: List[int],
    val_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Split data into train and validation sets.
    
    Args:
        sentences: All sentences
        labels: All labels
        val_ratio: Validation set ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_sentences, train_labels, val_sentences, val_labels)
    """
    random.seed(random_state)
    
    indices = list(range(len(sentences)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - val_ratio))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_sentences = [sentences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_sentences = [sentences[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_sentences, train_labels, val_sentences, val_labels


def create_data_loaders(
    sentences: List[str],
    labels: List[int],
    batch_size: int = 32,
    val_ratio: float = 0.1,
    max_length: int = MAX_SEQ_LENGTH,
    task: str = "esg_topic"
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation DataLoaders from sentences and labels.
    
    Args:
        sentences: List of sentence/paragraph texts
        labels: List of integer labels
        batch_size: Batch size
        val_ratio: Validation set ratio (0 = no validation set)
        max_length: Maximum sequence length (default 512 for paragraphs)
        task: Task type
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if val_ratio > 0:
        train_sentences, train_labels, val_sentences, val_labels = create_train_val_split(
            sentences, labels, val_ratio
        )
        
        train_loader = create_esg_dataloader(
            train_sentences, train_labels,
            batch_size=batch_size, shuffle=True,
            max_length=max_length, task=task
        )
        
        val_loader = create_esg_dataloader(
            val_sentences, val_labels,
            batch_size=batch_size, shuffle=False,
            max_length=max_length, task=task
        )
        
        return train_loader, val_loader
    else:
        train_loader = create_esg_dataloader(
            sentences, labels,
            batch_size=batch_size, shuffle=True,
            max_length=max_length, task=task
        )
        return train_loader, None
