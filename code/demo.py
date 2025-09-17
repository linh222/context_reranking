"""
Demonstration Script for Lifelog Context Retrieval
CBMI 2025

This script provides a complete demonstration of the lifelog context retrieval system,
showing how to use different components and methods for real-world scenarios.
"""

import os
import json
import pandas as pd
from typing import List, Dict
import logging

from lifelog_retrieval import TextualRetriever, VisualEncoder, MultimodalRetriever, RerankerModel
from train_models import RerankerTrainer
from evaluate_models import RetrievalEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LifelogDemo:
    """Demonstration class for the lifelog retrieval system."""
    
    def __init__(self):
        self.text_retriever = None
        self.visual_encoder = None
        self.multimodal_retriever = None
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, List]:
        """Create sample lifelog data for demonstration."""
        
        return {
            'contexts': [
                "Had breakfast at 8 AM with scrambled eggs, toast, and orange juice in the kitchen",
                "Attended a team meeting at 10 AM in the conference room to discuss project deadlines",
                "Went for a 30-minute walk in Central Park during lunch break, sunny weather",
                "Had lunch at Italian restaurant with pasta and salad, met with client",
                "Worked on Python code for data analysis project from 2-5 PM",
                "Visited grocery store after work, bought milk, bread, and vegetables",
                "Cooked dinner with family, made chicken curry and rice",
                "Watched a documentary about marine life on Netflix in the evening",
                "Read a book about machine learning before going to bed",
                "Had a video call with friends to plan weekend trip"
            ],
            'queries': [
                "What did I eat for breakfast?",
                "When was my team meeting?",
                "Where did I go during lunch?",
                "What restaurant did I visit?",
                "What programming work did I do?",
                "What groceries did I buy?",
                "What did I cook for dinner?",
                "What show did I watch?",
                "What book was I reading?",
                "What plans did I make with friends?"
            ],
            'image_paths': []  # In a real scenario, these would be actual image paths
        }
    
    def setup_retrievers(self):
        """Initialize all retrieval components."""
        
        logger.info("Setting up retrieval components...")
        
        # Initialize text retriever
        self.text_retriever = TextualRetriever("bert-base-uncased")
        logger.info("✓ Text retriever initialized")
        
        # Initialize visual encoder
        try:
            self.visual_encoder = VisualEncoder("clip")
            logger.info("✓ Visual encoder initialized")
        except Exception as e:
            logger.warning(f"Could not initialize visual encoder: {e}")
            self.visual_encoder = None
        
        # Initialize multimodal retriever
        if self.visual_encoder:
            self.multimodal_retriever = MultimodalRetriever(
                self.text_retriever, 
                self.visual_encoder
            )
            logger.info("✓ Multimodal retriever initialized")
        
    def demonstrate_text_retrieval(self):
        """Demonstrate text-based retrieval."""
        
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING TEXT-BASED RETRIEVAL")
        logger.info("="*60)
        
        if not self.text_retriever:
            logger.error("Text retriever not initialized!")
            return
        
        for i, query in enumerate(self.sample_data['queries'][:3]):  # Show first 3 queries
            logger.info(f"\nQuery {i+1}: {query}")
            logger.info("-" * 50)
            
            # Retrieve contexts
            results = self.text_retriever.retrieve_contexts(
                query, 
                self.sample_data['contexts'], 
                top_k=3
            )
            
            for j, (context, score) in enumerate(results, 1):
                logger.info(f"  {j}. Score: {score:.4f}")
                logger.info(f"     Context: {context}")
    
    def demonstrate_multimodal_retrieval(self):
        """Demonstrate multimodal retrieval."""
        
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING MULTIMODAL RETRIEVAL") 
        logger.info("="*60)
        
        if not self.multimodal_retriever:
            logger.warning("Multimodal retriever not available!")
            return
        
        for i, query in enumerate(self.sample_data['queries'][:2]):  # Show first 2 queries
            logger.info(f"\nQuery {i+1}: {query}")
            logger.info("-" * 50)
            
            # Retrieve and rerank contexts
            results = self.multimodal_retriever.retrieve_and_rerank(
                query,
                self.sample_data['contexts'],
                self.sample_data['image_paths'],
                top_k=3
            )
            
            for j, result in enumerate(results, 1):
                logger.info(f"  {j}. Combined Score: {result['combined_score']:.4f}")
                logger.info(f"     Text Score: {result['text_score']:.4f}")
                logger.info(f"     Visual Score: {result['visual_score']:.4f}")
                logger.info(f"     Context: {result['context']}")
    
    def demonstrate_training_pipeline(self):
        """Demonstrate the training pipeline."""
        
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING TRAINING PIPELINE")
        logger.info("="*60)
        
        # Create sample training data
        training_data = []
        for i, (query, context) in enumerate(zip(self.sample_data['queries'], self.sample_data['contexts'])):
            training_data.append({
                'question': query,
                'context': context,
                'label': 1  # Positive example
            })
            
            # Add negative examples
            for j, neg_context in enumerate(self.sample_data['contexts']):
                if i != j:  # Different context
                    training_data.append({
                        'question': query,
                        'context': neg_context,
                        'label': 0  # Negative example
                    })
        
        logger.info(f"Created {len(training_data)} training samples")
        
        # Save sample data for training
        sample_df = pd.DataFrame(training_data)
        sample_data_path = "sample_training_data.csv"
        sample_df.to_csv(sample_data_path, index=False)
        logger.info(f"Sample training data saved to {sample_data_path}")
        
        # Note: Actual training would require more data and computational resources
        logger.info("Note: For actual training, use the train_models.py script with larger datasets")
    
    def demonstrate_evaluation(self):
        """Demonstrate the evaluation process."""
        
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING EVALUATION PROCESS")
        logger.info("="*60)
        
        # Create test data
        test_data = []
        for query, context in zip(self.sample_data['queries'][:5], self.sample_data['contexts'][:5]):
            test_data.append({
                'question': query,
                'context': context,
                'label': 1
            })
        
        # Initialize evaluator
        evaluator = RetrievalEvaluator(test_data)
        
        # Evaluate text retriever
        if self.text_retriever:
            metrics = evaluator.evaluate_retrieval_method("BERT-Text", self.text_retriever)
            
            logger.info("Text Retrieval Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Compare methods if multimodal retriever is available
        if self.multimodal_retriever:
            methods = {
                "Text-Only": self.text_retriever,
                "Multimodal": self.multimodal_retriever
            }
            
            logger.info("\nComparing retrieval methods...")
            results_df = evaluator.compare_methods(methods)
            logger.info("\nComparison Results:")
            print(results_df.round(4))
    
    def interactive_demo(self):
        """Run an interactive demonstration."""
        
        logger.info("\n" + "="*60)
        logger.info("INTERACTIVE DEMONSTRATION")
        logger.info("="*60)
        
        if not self.text_retriever:
            logger.error("Retrievers not initialized!")
            return
        
        logger.info("Enter your own queries to search through the sample lifelog data!")
        logger.info("Available contexts:")
        for i, context in enumerate(self.sample_data['contexts'], 1):
            logger.info(f"  {i}. {context}")
        
        while True:
            try:
                user_query = input("\nEnter your query (or 'quit' to exit): ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_query:
                    continue
                
                # Retrieve contexts
                results = self.text_retriever.retrieve_contexts(
                    user_query,
                    self.sample_data['contexts'],
                    top_k=3
                )
                
                logger.info(f"\nTop 3 results for: '{user_query}'")
                logger.info("-" * 50)
                
                for i, (context, score) in enumerate(results, 1):
                    logger.info(f"{i}. Score: {score:.4f}")
                    logger.info(f"   {context}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
        
        logger.info("Interactive demo ended.")
    
    def run_complete_demo(self):
        """Run the complete demonstration."""
        
        logger.info("🚀 Starting Lifelog Context Retrieval Demonstration")
        logger.info("="*80)
        
        # Setup
        self.setup_retrievers()
        
        # Demonstrate different components
        self.demonstrate_text_retrieval()
        self.demonstrate_multimodal_retrieval() 
        self.demonstrate_training_pipeline()
        self.demonstrate_evaluation()
        
        # Interactive demo
        try:
            self.interactive_demo()
        except:
            logger.info("Skipping interactive demo")
        
        logger.info("\n🎉 Demonstration completed!")
        logger.info("\nNext steps:")
        logger.info("1. Prepare your own lifelog dataset")
        logger.info("2. Use train_models.py to train on your data")
        logger.info("3. Use evaluate_models.py to evaluate performance")
        logger.info("4. Integrate the system into your lifelog application")


def main():
    """Main demonstration function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Lifelog Context Retrieval Demo")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive demo only")
    parser.add_argument("--component", type=str, choices=["text", "multimodal", "training", "evaluation"],
                       help="Run specific component demo")
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = LifelogDemo()
    
    if args.interactive:
        demo.setup_retrievers()
        demo.interactive_demo()
    elif args.component:
        demo.setup_retrievers()
        if args.component == "text":
            demo.demonstrate_text_retrieval()
        elif args.component == "multimodal":
            demo.demonstrate_multimodal_retrieval()
        elif args.component == "training":
            demo.demonstrate_training_pipeline()
        elif args.component == "evaluation":
            demo.demonstrate_evaluation()
    else:
        # Run complete demo
        demo.run_complete_demo()


if __name__ == "__main__":
    main()