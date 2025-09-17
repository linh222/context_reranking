"""
Evaluation Script for Lifelog Context Retrieval
CBMI 2025

This script evaluates the performance of different retrieval methods
and provides comprehensive metrics for the paper results.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, ndcg_score
import torch
from tqdm import tqdm
import logging

from lifelog_retrieval import (
    TextualRetriever, VisualEncoder, MultimodalRetriever, 
    RerankerModel, load_lifelog_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Comprehensive evaluation class for lifelog retrieval systems."""
    
    def __init__(self, ground_truth_data: List[Dict]):
        self.ground_truth = ground_truth_data
        self.results = {}
    
    def calculate_mrr(self, retrieved_items: List[str], relevant_items: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, item in enumerate(retrieved_items):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_recall_at_k(self, retrieved_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Recall@K."""
        retrieved_k = retrieved_items[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_items))
        return relevant_retrieved / len(relevant_items) if relevant_items else 0.0
    
    def calculate_precision_at_k(self, retrieved_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Precision@K."""
        retrieved_k = retrieved_items[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_items))
        return relevant_retrieved / k if k > 0 else 0.0
    
    def calculate_ndcg_at_k(self, retrieved_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate NDCG@K."""
        retrieved_k = retrieved_items[:k]
        relevance_scores = [1 if item in relevant_items else 0 for item in retrieved_k]
        
        # Pad with zeros if needed
        while len(relevance_scores) < k:
            relevance_scores.append(0)
        
        # Calculate NDCG
        try:
            return ndcg_score([relevance_scores], [relevance_scores], k=k)
        except:
            return 0.0
    
    def evaluate_retrieval_method(self, method_name: str, retriever, top_k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Evaluate a specific retrieval method."""
        
        logger.info(f"Evaluating {method_name}...")
        
        metrics = {f'MRR': [], f'MAP': []}
        for k in top_k_values:
            metrics[f'Recall@{k}'] = []
            metrics[f'Precision@{k}'] = []
            metrics[f'NDCG@{k}'] = []
        
        all_queries = []
        all_contexts = []
        all_relevant = []
        
        # Prepare data
        for item in self.ground_truth:
            all_queries.append(item['question'])
            all_contexts.append(item['context'])
            # In real scenario, you'd have multiple relevant contexts per query
            all_relevant.append([item['context']])  # Simplified: only one relevant context
        
        # Evaluate each query
        for i, (query, relevant_contexts) in enumerate(tqdm(zip(all_queries, all_relevant), 
                                                             desc=f"Evaluating {method_name}")):
            
            try:
                # Get retrieval results
                if hasattr(retriever, 'retrieve_and_rerank'):
                    # Multimodal retriever
                    results = retriever.retrieve_and_rerank(query, all_contexts, [], top_k=max(top_k_values))
                    retrieved_contexts = [result['context'] for result in results]
                else:
                    # Text-only retriever
                    results = retriever.retrieve_contexts(query, all_contexts, top_k=max(top_k_values))
                    retrieved_contexts = [result[0] for result in results]
                
                # Calculate metrics
                mrr = self.calculate_mrr(retrieved_contexts, relevant_contexts)
                metrics['MRR'].append(mrr)
                
                # Average Precision (simplified)
                ap = average_precision_score(
                    [1 if ctx in relevant_contexts else 0 for ctx in retrieved_contexts[:max(top_k_values)]],
                    [1.0] * len(retrieved_contexts[:max(top_k_values)])  # Assuming equal relevance
                ) if retrieved_contexts else 0.0
                metrics['MAP'].append(ap)
                
                # Calculate metrics for different k values
                for k in top_k_values:
                    recall_k = self.calculate_recall_at_k(retrieved_contexts, relevant_contexts, k)
                    precision_k = self.calculate_precision_at_k(retrieved_contexts, relevant_contexts, k)
                    ndcg_k = self.calculate_ndcg_at_k(retrieved_contexts, relevant_contexts, k)
                    
                    metrics[f'Recall@{k}'].append(recall_k)
                    metrics[f'Precision@{k}'].append(precision_k)
                    metrics[f'NDCG@{k}'].append(ndcg_k)
            
            except Exception as e:
                logger.warning(f"Error evaluating query {i}: {e}")
                # Add zero scores for failed queries
                for metric_list in metrics.values():
                    metric_list.append(0.0)
        
        # Calculate average metrics
        avg_metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}
        
        self.results[method_name] = avg_metrics
        return avg_metrics
    
    def compare_methods(self, methods: Dict[str, object], save_path: Optional[str] = None) -> pd.DataFrame:
        """Compare multiple retrieval methods."""
        
        comparison_results = {}
        
        for method_name, retriever in methods.items():
            metrics = self.evaluate_retrieval_method(method_name, retriever)
            comparison_results[method_name] = metrics
        
        # Create comparison DataFrame
        df_results = pd.DataFrame(comparison_results).T
        
        # Save results
        if save_path:
            df_results.to_csv(save_path)
            logger.info(f"Results saved to {save_path}")
        
        return df_results
    
    def plot_results(self, save_dir: str = "results/"):
        """Generate plots for the evaluation results."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results:
            logger.warning("No results to plot. Run evaluation first.")
            return
        
        # Plot 1: Recall@K comparison
        plt.figure(figsize=(12, 8))
        
        k_values = [1, 3, 5, 10]
        methods = list(self.results.keys())
        
        for method in methods:
            recall_values = [self.results[method].get(f'Recall@{k}', 0) for k in k_values]
            plt.plot(k_values, recall_values, marker='o', linewidth=2, label=method)
        
        plt.xlabel('K')
        plt.ylabel('Recall@K')
        plt.title('Recall@K Comparison Across Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'recall_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Overall metrics heatmap
        metrics_for_heatmap = ['MRR', 'MAP', 'Recall@5', 'Precision@5', 'NDCG@5']
        heatmap_data = []
        
        for method in methods:
            method_data = [self.results[method].get(metric, 0) for metric in metrics_for_heatmap]
            heatmap_data.append(method_data)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, 
                   xticklabels=metrics_for_heatmap,
                   yticklabels=methods,
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues')
        plt.title('Performance Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_dir}")


def load_test_data(data_path: str) -> List[Dict]:
    """Load test data for evaluation."""
    return load_lifelog_data(data_path)


def create_baseline_methods() -> Dict[str, object]:
    """Create baseline retrieval methods for comparison."""
    
    methods = {}
    
    # Text-only retrieval with BERT
    methods['BERT-Text'] = TextualRetriever("bert-base-uncased")
    
    # Text-only retrieval with different model
    methods['DistilBERT-Text'] = TextualRetriever("distilbert-base-uncased")
    
    # Multimodal retrieval
    text_retriever = TextualRetriever("bert-base-uncased")
    visual_encoder = VisualEncoder("clip")
    methods['Multimodal-BERT-CLIP'] = MultimodalRetriever(text_retriever, visual_encoder)
    
    return methods


def load_trained_models(model_dir: str) -> Dict[str, object]:
    """Load trained models for evaluation."""
    
    methods = {}
    
    # Load reranker model if available
    reranker_path = os.path.join(model_dir, 'best_reranker_model.pth')
    if os.path.exists(reranker_path):
        reranker = RerankerModel("bert-base-uncased")
        checkpoint = torch.load(reranker_path, map_location='cpu')
        reranker.load_state_dict(checkpoint['model_state_dict'])
        
        # Create multimodal retriever with reranker
        text_retriever = TextualRetriever("bert-base-uncased")
        visual_encoder = VisualEncoder("clip")
        methods['Multimodal-BERT-CLIP-Reranker'] = MultimodalRetriever(
            text_retriever, visual_encoder, reranker
        )
    
    return methods


def run_evaluation(test_data_path: str, model_dir: str = "models/", results_dir: str = "results/"):
    """Run comprehensive evaluation."""
    
    logger.info("Starting comprehensive evaluation...")
    
    # Load test data
    test_data = load_test_data(test_data_path)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    if not test_data:
        logger.error("No test data loaded!")
        return
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(test_data)
    
    # Create baseline methods
    baseline_methods = create_baseline_methods()
    
    # Load trained models
    trained_methods = load_trained_models(model_dir)
    
    # Combine all methods
    all_methods = {**baseline_methods, **trained_methods}
    
    logger.info(f"Evaluating {len(all_methods)} methods: {list(all_methods.keys())}")
    
    # Run comparison
    results_df = evaluator.compare_methods(all_methods, 
                                          save_path=os.path.join(results_dir, 'evaluation_results.csv'))
    
    # Generate plots
    evaluator.plot_results(results_dir)
    
    # Print summary
    logger.info("\nEvaluation Results Summary:")
    logger.info("=" * 50)
    print(results_df.round(4))
    
    # Find best method for each metric
    logger.info("\nBest Methods by Metric:")
    logger.info("-" * 30)
    for metric in results_df.columns:
        best_method = results_df[metric].idxmax()
        best_score = results_df[metric].max()
        logger.info(f"{metric}: {best_method} ({best_score:.4f})")
    
    return results_df


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Lifelog Retrieval Models")
    parser.add_argument("--test_data", type=str, required=True, 
                       help="Path to test data CSV")
    parser.add_argument("--model_dir", type=str, default="models/", 
                       help="Directory containing trained models")
    parser.add_argument("--results_dir", type=str, default="results/", 
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Run evaluation
    results_df = run_evaluation(args.test_data, args.model_dir, args.results_dir)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()