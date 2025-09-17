"""
Lifelog Context Retrieval Implementation
CBMI 2025

This module implements the core methods for lifelog context retrieval,
including text-based retrieval, visual embedding, and reranking models.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
from typing import List, Dict, Tuple, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LifelogDataset(Dataset):
    """Dataset class for lifelog question-answering pairs with context."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine question and context
        text = f"Question: {item['question']} Context: {item['context']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(item.get('label', 0), dtype=torch.long)
        }


class TextualRetriever:
    """Text-based retrieval using BERT-based models."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding)
        
        return np.vstack(embeddings)
    
    def retrieve_contexts(self, query: str, contexts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k most similar contexts for a given query."""
        query_embedding = self.encode_text([query])
        context_embeddings = self.encode_text(contexts)
        
        # Compute cosine similarity
        similarities = np.dot(query_embedding, context_embeddings.T).flatten()
        similarities = similarities / (np.linalg.norm(query_embedding) * np.linalg.norm(context_embeddings, axis=1))
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [(contexts[idx], similarities[idx]) for idx in top_indices]
        return results


class VisualEncoder:
    """Visual encoder using CLIP or BLIP models."""
    
    def __init__(self, model_type: str = "clip"):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == "clip":
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode images into visual embeddings."""
        embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for image_path in image_paths:
                try:
                    image = Image.open(image_path)
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    if self.model_type == "clip":
                        embedding = self.model.encode_image(image_input).cpu().numpy()
                    
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    # Add zero embedding for failed images
                    embeddings.append(np.zeros((1, 512)))
        
        return np.vstack(embeddings)
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts using visual model's text encoder."""
        embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                text_tokens = clip.tokenize([text]).to(self.device)
                embedding = self.model.encode_text(text_tokens).cpu().numpy()
                embeddings.append(embedding)
        
        return np.vstack(embeddings)


class RerankerModel(nn.Module):
    """Neural reranking model for lifelog retrieval."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2):
        super(RerankerModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class MultimodalRetriever:
    """Multimodal retrieval combining textual and visual information."""
    
    def __init__(self, text_retriever: TextualRetriever, visual_encoder: VisualEncoder, reranker: Optional[RerankerModel] = None):
        self.text_retriever = text_retriever
        self.visual_encoder = visual_encoder
        self.reranker = reranker
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.reranker:
            self.reranker.to(self.device)
    
    def retrieve_and_rerank(self, query: str, contexts: List[str], image_paths: List[str], top_k: int = 5) -> List[Dict]:
        """Retrieve contexts using multimodal information and rerank them."""
        
        # Step 1: Text-based retrieval
        text_results = self.text_retriever.retrieve_contexts(query, contexts, top_k * 2)
        
        # Step 2: Visual similarity (if images available)
        if image_paths and len(image_paths) == len(contexts):
            query_text_embedding = self.visual_encoder.encode_text([query])
            image_embeddings = self.visual_encoder.encode_images(image_paths)
            
            # Compute text-image similarity
            visual_similarities = np.dot(query_text_embedding, image_embeddings.T).flatten()
            visual_similarities = visual_similarities / (np.linalg.norm(query_text_embedding) * np.linalg.norm(image_embeddings, axis=1))
        else:
            visual_similarities = np.zeros(len(contexts))
        
        # Step 3: Combine scores
        combined_results = []
        for context, text_score in text_results:
            context_idx = contexts.index(context)
            visual_score = visual_similarities[context_idx] if len(visual_similarities) > context_idx else 0.0
            
            # Weighted combination
            combined_score = 0.7 * text_score + 0.3 * visual_score
            
            combined_results.append({
                'context': context,
                'text_score': text_score,
                'visual_score': visual_score,
                'combined_score': combined_score,
                'image_path': image_paths[context_idx] if image_paths and len(image_paths) > context_idx else None
            })
        
        # Step 4: Reranking (if reranker available)
        if self.reranker:
            combined_results = self._rerank_results(query, combined_results)
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return combined_results[:top_k]
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Apply neural reranking to the results."""
        self.reranker.eval()
        
        with torch.no_grad():
            for result in results:
                # Prepare input for reranker
                text = f"Query: {query} Context: {result['context']}"
                inputs = self.text_retriever.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Get reranking score
                logits = self.reranker(inputs['input_ids'], inputs['attention_mask'])
                rerank_score = torch.softmax(logits, dim=1)[0, 1].item()  # Positive class probability
                
                # Update combined score with reranking
                result['rerank_score'] = rerank_score
                result['combined_score'] = 0.5 * result['combined_score'] + 0.5 * rerank_score
        
        return results


def load_lifelog_data(data_path: str) -> List[Dict]:
    """Load lifelog data from CSV file."""
    try:
        df = pd.read_csv(data_path)
        data = []
        
        for _, row in df.iterrows():
            data.append({
                'question': row.get('question', ''),
                'context': row.get('context', ''),
                'image_id': row.get('ImageID', ''),
                'label': row.get('label', 0)
            })
        
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return []


def evaluate_retrieval(retriever: MultimodalRetriever, test_data: List[Dict], top_k: int = 5) -> Dict[str, float]:
    """Evaluate retrieval performance."""
    correct = 0
    total = 0
    
    for item in test_data:
        query = item['question']
        # This is a simplified evaluation - in practice, you'd have ground truth contexts
        # and measure metrics like MRR, NDCG, etc.
        
        # Placeholder evaluation logic
        total += 1
        # In real implementation, check if correct context is in top-k results
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'total_queries': total
    }


# Example usage
if __name__ == "__main__":
    # Initialize components
    text_retriever = TextualRetriever("bert-base-uncased")
    visual_encoder = VisualEncoder("clip")
    
    # Example data
    query = "What did I eat for lunch yesterday?"
    contexts = [
        "Had a delicious sandwich with turkey and cheese at the cafe",
        "Attended a meeting with colleagues about the project",
        "Went for a walk in the park after lunch"
    ]
    image_paths = []  # Add actual image paths here
    
    # Create multimodal retriever
    multimodal_retriever = MultimodalRetriever(text_retriever, visual_encoder)
    
    # Retrieve and rank results
    results = multimodal_retriever.retrieve_and_rerank(query, contexts, image_paths, top_k=3)
    
    # Display results
    print(f"Query: {query}")
    print("\nTop retrieved contexts:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Context: {result['context']}")
        print(f"   Combined Score: {result['combined_score']:.4f}")
        print(f"   Text Score: {result['text_score']:.4f}")
        print(f"   Visual Score: {result['visual_score']:.4f}")
        print()