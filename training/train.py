"""
EVINCE: Training Script

Training pipeline for EVINCE models:
- ESG Topic Classifier
- Washing Detector
- Sentence Type Classifier
- Claim-Evidence Matcher
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from tqdm import tqdm
import logging

try:
    from transformers import get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer for EVINCE models.
    
    Supports:
    - Mixed precision training
    - Learning rate scheduling
    - Gradient accumulation
    - Checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        device: str = "auto",
        output_dir: str = "./checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            num_epochs: Number of training epochs
            warmup_ratio: Warmup ratio for scheduler
            gradient_accumulation_steps: Gradient accumulation steps
            device: Training device
            output_dir: Directory for checkpoints
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        
        if TRANSFORMERS_AVAILABLE:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None
        
        # Loss function (default: CrossEntropy)
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.best_val_acc = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Handle models that return tuple (logits, attention)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            progress.set_postfix({"loss": total_loss / num_batches})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_acc": 0.0}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total
        
        return {"val_loss": val_loss, "val_acc": val_acc}
    
    @torch.no_grad()
    def evaluate_loader(self, data_loader) -> Dict[str, float]:
        """Evaluate on any DataLoader (for test set evaluation)."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in data_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        
        loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history
        }
        
        # Save latest
        path = os.path.join(self.output_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = os.path.join(self.output_dir, "checkpoint_best.pt")
            torch.save(checkpoint, path)
            logger.info(f"Saved best model with val_acc={self.best_val_acc:.4f}")
    
    def train(self) -> Dict[str, list]:
        """Run full training."""
        logger.info(f"Training on {self.device}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            
            # Evaluate
            val_metrics = self.evaluate()
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_acc"].append(val_metrics["val_acc"])
            
            logger.info(
                f"Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"val_acc={val_metrics['val_acc']:.4f}"
            )
            
            # Save checkpoint
            is_best = val_metrics["val_acc"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["val_acc"]
            self.save_checkpoint(epoch, is_best)
        
        logger.info(f"Training complete. Best val_acc: {self.best_val_acc:.4f}")
        return self.history


def train_esg_classifier(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    output_dir: str = "./checkpoints/esg_classifier"
) -> Trainer:
    """
    Train ESG Topic Classifier.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        output_dir: Output directory
        
    Returns:
        Trained Trainer instance
    """
    from ..models import ESGTopicClassifier
    
    model = ESGTopicClassifier(freeze_bert_layers=6)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir
    )
    
    trainer.train()
    return trainer


def train_washing_detector(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 15,
    learning_rate: float = 1e-5,
    output_dir: str = "./checkpoints/washing_detector"
) -> Trainer:
    """
    Train Washing Detector.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        output_dir: Output directory
        
    Returns:
        Trained Trainer instance
    """
    from ..models import WashingDetector
    
    model = WashingDetector(freeze_bert_layers=6)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir
    )
    
    trainer.train()
    return trainer
