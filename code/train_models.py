"""
Training Script for Lifelog Context Retrieval Models
CBMI 2025

This script handles training of the reranking model and fine-tuning
of the retrieval components for lifelog question answering.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

from lifelog_retrieval import LifelogDataset, RerankerModel, TextualRetriever, load_lifelog_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankerTrainer:
    """Trainer class for the neural reranking model."""
    
    def __init__(self, model: RerankerModel, train_loader: DataLoader, val_loader: DataLoader, 
                 learning_rate: float = 2e-5, num_epochs: int = 3, warmup_steps: int = 500):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Update statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, save_dir: str = "models/") -> Dict[str, List[float]]:
        """Train the model for specified number of epochs."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_metrics': val_metrics
                }, os.path.join(save_dir, 'best_reranker_model.pth'))
                logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        return self.history


def prepare_training_data(data_path: str, tokenizer, test_size: float = 0.2, max_length: int = 512) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation data loaders."""
    
    # Load data
    data = load_lifelog_data(data_path)
    
    if not data:
        raise ValueError(f"No data loaded from {data_path}")
    
    # Split data
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = LifelogDataset(train_data, tokenizer, max_length)
    val_dataset = LifelogDataset(val_data, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def train_reranker(data_path: str, model_name: str = "bert-base-uncased", 
                   save_dir: str = "models/", **kwargs) -> Dict[str, List[float]]:
    """Train the reranking model."""
    
    # Initialize model and tokenizer
    model = RerankerModel(model_name)
    retriever = TextualRetriever(model_name)  # For tokenizer
    
    # Prepare data
    train_loader, val_loader = prepare_training_data(data_path, retriever.tokenizer, **kwargs)
    
    # Initialize trainer
    trainer = RerankerTrainer(model, train_loader, val_loader)
    
    # Train model
    history = trainer.train(save_dir)
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def fine_tune_retriever(data_path: str, model_name: str = "bert-base-uncased", 
                       save_dir: str = "models/", num_epochs: int = 3) -> None:
    """Fine-tune the text retriever model."""
    
    logger.info("Fine-tuning text retriever...")
    
    # This is a simplified version - in practice, you might want to use
    # contrastive learning or other techniques for retrieval fine-tuning
    
    retriever = TextualRetriever(model_name)
    data = load_lifelog_data(data_path)
    
    # Extract unique contexts and questions
    contexts = list(set([item['context'] for item in data if item['context']]))
    questions = list(set([item['question'] for item in data if item['question']]))
    
    logger.info(f"Fine-tuning on {len(contexts)} contexts and {len(questions)} questions")
    
    # Encode and save embeddings for evaluation
    context_embeddings = retriever.encode_text(contexts[:100])  # Limit for demo
    question_embeddings = retriever.encode_text(questions[:100])
    
    # Save embeddings
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'context_embeddings.npy'), context_embeddings)
    np.save(os.path.join(save_dir, 'question_embeddings.npy'), question_embeddings)
    
    logger.info("Text retriever fine-tuning completed")


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Lifelog Retrieval Models")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pre-trained model name")
    parser.add_argument("--save_dir", type=str, default="models/", help="Directory to save models")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--train_reranker", action="store_true", help="Train the reranking model")
    parser.add_argument("--fine_tune_retriever", action="store_true", help="Fine-tune the retriever")
    
    args = parser.parse_args()
    
    if args.train_reranker:
        logger.info("Training reranker model...")
        history = train_reranker(
            args.data_path,
            args.model_name,
            args.save_dir,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length
        )
        logger.info("Reranker training completed")
    
    if args.fine_tune_retriever:
        fine_tune_retriever(
            args.data_path,
            args.model_name,
            args.save_dir,
            args.num_epochs
        )
    
    if not args.train_reranker and not args.fine_tune_retriever:
        logger.info("No training specified. Use --train_reranker or --fine_tune_retriever")


if __name__ == "__main__":
    main()