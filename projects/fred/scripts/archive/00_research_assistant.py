#!/usr/bin/env python3
"""
FRED Research Assistant
Takes natural language research questions and orchestrates the entire data pipeline:
1. Analyzes the query to extract relevant search terms
2. Fetches data from FRED API
3. Parses and processes the data
4. Creates embeddings for semantic search
5. Integrates with Kumo RFM for predictions
6. Saves to monolith structure

Example:
    python 00_research_assistant.py "I would like to research money supply"
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Try to import LLM libraries for dynamic concept extraction
try:
    from transformers import pipeline
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Knowledge base of economic concepts and their related FRED search terms
ECONOMIC_CONCEPTS = {
    'money supply': ['money supply', 'M1', 'M2', 'M3', 'monetary base', 'bank reserves', 'currency in circulation'],
    'inflation': ['inflation', 'CPI', 'PPI', 'PCE', 'deflator', 'core inflation'],
    'employment': ['employment', 'unemployment', 'labor force', 'jobs', 'payroll', 'wages', 'labor participation'],
    'interest rates': ['interest rate', 'federal funds rate', 'treasury yield', 'mortgage rate', 'prime rate', 'libor'],
    'gdp': ['GDP', 'gross domestic product', 'economic growth', 'real GDP', 'nominal GDP'],
    'housing': ['housing starts', 'home sales', 'home prices', 'building permits', 'Case-Shiller', 'mortgage', 'rent'],
    'trade': ['exports', 'imports', 'trade balance', 'trade deficit', 'current account', 'exchange rate'],
    'consumer': ['consumer spending', 'retail sales', 'consumer confidence', 'personal income', 'disposable income', 'savings rate'],
    'business': ['industrial production', 'capacity utilization', 'manufacturing', 'durable goods', 'inventory', 'productivity'],
    'debt': ['household debt', 'corporate debt', 'government debt', 'deficit', 'federal debt', 'credit'],
    'markets': ['stock market', 'S&P 500', 'volatility', 'corporate profits', 'dividend yield', 'equity'],
    'commodities': ['oil prices', 'energy prices', 'commodity prices', 'food prices', 'gas prices', 'gold prices'],
    'banking': ['bank reserves', 'credit', 'lending', 'deposits', 'commercial banks', 'bank assets'],
    'prices': ['prices', 'price index', 'cost of living', 'purchasing power'],
    'productivity': ['productivity', 'output', 'efficiency', 'labor productivity', 'total factor productivity'],
}

# Common keywords to help identify concepts
KEYWORD_MAPPING = {
    'monetary': 'money supply',
    'cash': 'money supply',
    'liquidity': 'money supply',
    'fed': 'interest rates',
    'central bank': 'interest rates',
    'rates': 'interest rates',
    'jobs': 'employment',
    'unemployment': 'employment',
    'labor': 'employment',
    'workers': 'employment',
    'price': 'inflation',
    'cost': 'inflation',
    'housing': 'housing',
    'real estate': 'housing',
    'homes': 'housing',
    'trade': 'trade',
    'export': 'trade',
    'import': 'trade',
    'consumer': 'consumer',
    'retail': 'consumer',
    'spending': 'consumer',
    'stocks': 'markets',
    'equity': 'markets',
    'market': 'markets',
    'oil': 'commodities',
    'energy': 'commodities',
    'commodity': 'commodities',
}


class EconomicConceptExtractor:
    """
    Extracts economic concepts and search terms from natural language queries.
    Supports both rule-based (static) and LLM-based (dynamic) extraction.
    """
    
    def __init__(self, use_llm: bool = False, llm_model: str = "facebook/bart-large-mnli"):
        """
        Initialize the concept extractor.
        
        Args:
            use_llm: Whether to use LLM-based extraction (requires transformers)
            llm_model: HuggingFace model to use for zero-shot classification
        """
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_model = None
        
        if self.use_llm:
            print(f"Loading LLM model: {llm_model}...")
            try:
                self.llm_model = pipeline("zero-shot-classification", model=llm_model, device=-1)
                print("LLM model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load LLM model: {e}")
                print("Falling back to rule-based extraction")
                self.use_llm = False
    
    @staticmethod
    def extract_static(query: str) -> Tuple[List[str], str]:
        """
        Rule-based extraction using static ECONOMIC_CONCEPTS dictionary.
        
        Args:
            query: Natural language research question
            
        Returns:
            Tuple of (search_terms, category_description)
        """
        query_lower = query.lower()
        
        # Find matching concepts
        matched_concepts = []
        
        # First check direct concept matches
        for concept, terms in ECONOMIC_CONCEPTS.items():
            if concept in query_lower:
                matched_concepts.append(concept)
        
        # Then check keyword mappings
        for keyword, concept in KEYWORD_MAPPING.items():
            if keyword in query_lower and concept not in matched_concepts:
                matched_concepts.append(concept)
        
        # If no matches, try to extract key economic terms
        if not matched_concepts:
            # Look for any terms that appear in our knowledge base
            for concept, terms in ECONOMIC_CONCEPTS.items():
                for term in terms:
                    if term.lower() in query_lower:
                        matched_concepts.append(concept)
                        break
        
        # Get search terms for matched concepts
        search_terms = []
        for concept in matched_concepts:
            search_terms.extend(ECONOMIC_CONCEPTS.get(concept, []))
        
        # Remove duplicates while preserving order
        search_terms = list(dict.fromkeys(search_terms))
        
        # Create category description
        if matched_concepts:
            category = '_'.join(matched_concepts)
        else:
            # Fallback: use key words from query
            category = '_'.join([w for w in query_lower.split() if len(w) > 3][:3])
        
        return search_terms, category
    
    def extract_llm(self, query: str, top_k: int = 3) -> Tuple[List[str], str]:
        """
        LLM-based extraction using zero-shot classification.
        
        This method uses a small language model trained on natural language understanding
        to classify the query into economic concepts, then maps those to search terms.
        
        Args:
            query: Natural language research question
            top_k: Number of top concepts to extract
            
        Returns:
            Tuple of (search_terms, category_description)
        """
        if not self.llm_model:
            print("LLM model not available, falling back to static extraction")
            return self.extract_static(query)
        
        # Use zero-shot classification to identify relevant concepts
        candidate_labels = list(ECONOMIC_CONCEPTS.keys())
        
        try:
            result = self.llm_model(query, candidate_labels, multi_label=True)
            
            # Get top-k concepts with confidence > 0.3
            matched_concepts = []
            for label, score in zip(result['labels'], result['scores']):
                if score > 0.3 and len(matched_concepts) < top_k:
                    matched_concepts.append(label)
            
            # Get search terms for matched concepts
            search_terms = []
            for concept in matched_concepts:
                search_terms.extend(ECONOMIC_CONCEPTS.get(concept, []))
            
            # Remove duplicates while preserving order
            search_terms = list(dict.fromkeys(search_terms))
            
            # Create category
            category = '_'.join(matched_concepts) if matched_concepts else 'general'
            
            return search_terms, category
            
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            print("Falling back to static extraction")
            return self.extract_static(query)
    
    def extract(self, query: str) -> Tuple[List[str], str]:
        """
        Extract concepts using configured method (LLM or static).
        
        Args:
            query: Natural language research question
            
        Returns:
            Tuple of (search_terms, category_description)
        """
        if self.use_llm:
            return self.extract_llm(query)
        else:
            return self.extract_static(query)


# ============================================================================
# FUTURE: LLM-BASED DYNAMIC EXTRACTION WITH FINE-TUNING
# ============================================================================
# 
# The code below is a template for training a custom small LLM on economic
# concepts to improve concept extraction. Uncomment and modify as needed.
#
# Requirements:
#   pip install transformers datasets torch
#
# Training data format (JSON):
# [
#   {
#     "query": "I want to study the relationship between money supply and inflation",
#     "concepts": ["money supply", "inflation"],
#     "search_terms": ["M1", "M2", "CPI", "inflation", "money supply"]
#   },
#   ...
# ]
# ============================================================================

"""
class EconomicConceptTrainer:
    '''
    Train a small LLM to extract economic concepts from natural language queries.
    
    This uses a lightweight model (e.g., DistilBERT) fine-tuned on economic text
    to classify queries into relevant economic concepts.
    '''
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        '''
        Initialize the trainer.
        
        Args:
            model_name: Base model to fine-tune
        '''
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import Trainer, TrainingArguments
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
    
    def prepare_training_data(self, data_file: str):
        '''
        Prepare training data from annotated queries.
        
        Args:
            data_file: Path to JSON file with training examples
        '''
        import json
        from datasets import Dataset
        
        # Load training data
        with open(data_file) as f:
            data = json.load(f)
        
        # Convert to multi-label classification format
        # Each example can belong to multiple economic concepts
        examples = []
        for item in data:
            query = item['query']
            concepts = item['concepts']
            
            # Create binary labels for each concept
            labels = [1 if c in concepts else 0 for c in ECONOMIC_CONCEPTS.keys()]
            
            examples.append({
                'text': query,
                'labels': labels
            })
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True)
        
        dataset = Dataset.from_list(examples)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, train_dataset, output_dir: str = 'models/economic_concepts',
              num_epochs: int = 3, batch_size: int = 8):
        '''
        Train the model on annotated economic queries.
        
        Args:
            train_dataset: Prepared training dataset
            output_dir: Directory to save the trained model
            num_epochs: Number of training epochs
            batch_size: Batch size for training
        '''
        from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
        
        # Initialize model
        num_labels = len(ECONOMIC_CONCEPTS)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            save_strategy="epoch",
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train
        print(f"Training model on {len(train_dataset)} examples...")
        self.trainer.train()
        
        # Save
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def predict(self, query: str, threshold: float = 0.5) -> List[str]:
        '''
        Predict economic concepts from a query using the trained model.
        
        Args:
            query: Natural language research question
            threshold: Confidence threshold for concept prediction
            
        Returns:
            List of predicted concepts
        '''
        import torch
        
        # Tokenize input
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs.logits)
        
        # Extract concepts above threshold
        concepts = list(ECONOMIC_CONCEPTS.keys())
        predicted_concepts = [
            concepts[i] for i, score in enumerate(predictions[0])
            if score > threshold
        ]
        
        return predicted_concepts


def create_training_data_template(output_file: str = 'data/concept_training_data.json'):
    '''
    Create a template for training data annotation.
    
    Users can fill this out with real queries and their associated concepts,
    then use it to train a custom model.
    '''
    template = [
        {
            "query": "I want to study the relationship between money supply and inflation",
            "concepts": ["money supply", "inflation"],
            "search_terms": ["M1", "M2", "M3", "CPI", "PPI", "inflation", "money supply"]
        },
        {
            "query": "How does unemployment affect GDP growth?",
            "concepts": ["employment", "gdp"],
            "search_terms": ["unemployment", "GDP", "economic growth", "employment"]
        },
        {
            "query": "What are the trends in housing market and mortgage rates?",
            "concepts": ["housing", "interest rates"],
            "search_terms": ["housing starts", "home prices", "mortgage rate", "mortgage"]
        },
        {
            "query": "Analyze the impact of federal reserve policy on interest rates",
            "concepts": ["interest rates", "money supply"],
            "search_terms": ["federal funds rate", "interest rate", "monetary base"]
        },
        # Add more examples here...
    ]
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Training data template created: {output_file}")
    print("Fill this out with your own examples and use --train to fine-tune the model")
"""


class FREDResearchAssistant:
    """Intelligent assistant for economic data research."""
    
    def __init__(self, project_dir: str = '.', use_llm: bool = False):
        self.project_dir = Path(project_dir)
        self.txt_dir = self.project_dir / 'txt'
        self.data_dir = self.project_dir / 'data'
        self.embeddings_dir = self.project_dir / 'embeddings'
        self.monolith_dir = self.project_dir / 'monolith'
        
        # Create directories if they don't exist
        for d in [self.txt_dir, self.data_dir, self.embeddings_dir, self.monolith_dir]:
            d.mkdir(exist_ok=True, parents=True)
        
        # Initialize concept extractor
        self.extractor = EconomicConceptExtractor(use_llm=use_llm)
    
    def analyze_query(self, query: str) -> Tuple[List[str], str]:
        """
        Analyze natural language query to extract relevant search terms.
        
        Args:
            query: Natural language research question
            
        Returns:
            Tuple of (search_terms, category_description)
        """
        return self.extractor.extract(query)
    
    def recommend_series(self, query: str) -> Dict:
        """
        Recommend datasets based on the research query.
        
        Args:
            query: Natural language research question
            
        Returns:
            Dictionary with recommendations
        """
        search_terms, category = self.analyze_query(query)
        
        recommendation = {
            'query': query,
            'category': category,
            'search_terms': search_terms,
            'num_terms': len(search_terms),
            'extraction_method': 'llm' if self.extractor.use_llm else 'static',
            'timestamp': datetime.now().isoformat()
        }
        
        return recommendation
    
    def run_pipeline(self, query: str, limit: int = 1000, 
                    skip_fetch: bool = False,
                    skip_parse: bool = False,
                    skip_embeddings: bool = False,
                    skip_monolith: bool = False) -> Dict:
        """
        Run the complete research pipeline for a query.
        
        Args:
            query: Natural language research question
            limit: Maximum series per search term
            skip_*: Skip specific pipeline stages
            
        Returns:
            Dictionary with pipeline results
        """
        print("=" * 70)
        print("FRED RESEARCH ASSISTANT")
        print("=" * 70)
        print(f"\nQuery: {query}")
        print(f"Extraction method: {'LLM-based' if self.extractor.use_llm else 'Rule-based'}")
        print()
        
        # Step 1: Analyze query and recommend datasets
        print("Step 1: Analyzing query and recommending datasets...")
        recommendation = self.recommend_series(query)
        
        print(f"  Category: {recommendation['category']}")
        print(f"  Found {recommendation['num_terms']} relevant search terms:")
        for term in recommendation['search_terms']:
            print(f"    - {term}")
        print()
        
        # Save recommendation
        rec_file = self.data_dir / f"research_{recommendation['category']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(rec_file, 'w') as f:
            json.dump(recommendation, f, indent=2)
        print(f"  Saved recommendation to: {rec_file.name}\n")
        
        results = {'recommendation': recommendation}
        
        # Step 2: Fetch data from FRED
        if not skip_fetch:
            print("Step 2: Fetching data from FRED API...")
            fetch_result = self._run_fetch(recommendation['search_terms'], limit)
            results['fetch'] = fetch_result
            print()
        else:
            print("Step 2: Skipping data fetch (--skip-fetch)\n")
        
        # Step 3: Parse and consolidate data
        if not skip_parse:
            print("Step 3: Parsing and consolidating data...")
            parse_result = self._run_parse()
            results['parse'] = parse_result
            print()
        else:
            print("Step 3: Skipping parse (--skip-parse)\n")
        
        # Step 4: Create embeddings
        if not skip_embeddings:
            print("Step 4: Creating embeddings for semantic search...")
            embedding_result = self._run_embeddings()
            results['embeddings'] = embedding_result
            print()
        else:
            print("Step 4: Skipping embeddings (--skip-embeddings)\n")
        
        # Step 5: Save to monolith structure
        if not skip_monolith:
            print("Step 5: Preparing monolith features...")
            monolith_result = self._run_monolith()
            results['monolith'] = monolith_result
            print()
        else:
            print("Step 5: Skipping monolith (--skip-monolith)\n")
        
        # Summary
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to:")
        print(f"  - Raw data: {self.txt_dir}")
        print(f"  - Processed data: {self.data_dir}")
        print(f"  - Embeddings: {self.embeddings_dir}")
        print(f"  - Monolith: {self.monolith_dir}")
        print()
        
        return results
    
    def _run_fetch(self, search_terms: List[str], limit: int) -> Dict:
        """Run the fetch script."""
        cmd = [
            sys.executable,
            '01_fetch_fred_data.py',
            '--search'
        ] + search_terms + [
            '--limit', str(limit)
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_dir, 
                                   capture_output=True, text=True, check=True)
            return {'success': True, 'output': result.stdout}
        except subprocess.CalledProcessError as e:
            print(f"  Error in fetch: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_parse(self) -> Dict:
        """Run the parse script."""
        cmd = [
            sys.executable,
            '02_parse_fred_txt.py',
            '--input', 'txt',
            '--output', 'data/fred_series_metadata.parquet'
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_dir,
                                   capture_output=True, text=True, check=True)
            return {'success': True, 'output': result.stdout}
        except subprocess.CalledProcessError as e:
            print(f"  Error in parse: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_embeddings(self) -> Dict:
        """Run the embedding creation script."""
        cmd = [
            sys.executable,
            '04_vector_search.py',
            '--data', 'data/fred_series_metadata.parquet',
            '--create',
            '--save', 'embeddings/'
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_dir,
                                   capture_output=True, text=True, check=True)
            return {'success': True, 'output': result.stdout}
        except subprocess.CalledProcessError as e:
            print(f"  Error in embeddings: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_monolith(self) -> Dict:
        """Run the monolith preparation script."""
        cmd = [
            sys.executable,
            '06_prepare_monolith_features.py',
            '--input', 'data/fred_series_metadata.parquet',
            '--output', 'monolith/'
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_dir,
                                   capture_output=True, text=True, check=True)
            return {'success': True, 'output': result.stdout}
        except subprocess.CalledProcessError as e:
            print(f"  Error in monolith: {e}")
            return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Intelligent FRED Research Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Research a topic and run full pipeline
  python 00_research_assistant.py "I would like to research money supply"
  
  # Use LLM-based concept extraction (requires transformers)
  python 00_research_assistant.py "inflation and price trends" --use-llm
  
  # Just see recommendations without fetching
  python 00_research_assistant.py "inflation and price trends" --recommend-only
  
  # Run with fewer series per term
  python 00_research_assistant.py "housing market analysis" --limit 100
  
  # Skip certain pipeline stages
  python 00_research_assistant.py "labor markets" --skip-embeddings --skip-monolith
  
  # List available economic concepts
  python 00_research_assistant.py --list-concepts
  
  # Create training data template for custom model
  python 00_research_assistant.py --create-training-template
        """
    )
    
    parser.add_argument('query', nargs='*', help='Natural language research question')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Max series per search term (default: 1000)')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM-based concept extraction (requires transformers)')
    parser.add_argument('--recommend-only', action='store_true',
                       help='Only show recommendations, do not fetch data')
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip data fetching (use existing data)')
    parser.add_argument('--skip-parse', action='store_true',
                       help='Skip parsing step')
    parser.add_argument('--skip-embeddings', action='store_true',
                       help='Skip embedding creation')
    parser.add_argument('--skip-monolith', action='store_true',
                       help='Skip monolith preparation')
    parser.add_argument('--list-concepts', action='store_true',
                       help='List available economic concepts')
    parser.add_argument('--create-training-template', action='store_true',
                       help='Create a template for LLM training data')
    parser.add_argument('--project-dir', type=str, default='.',
                       help='Project directory (default: current)')
    
    args = parser.parse_args()
    
    # List concepts
    if args.list_concepts:
        print("\nAvailable Economic Concepts:")
        print("=" * 70)
        for concept, terms in sorted(ECONOMIC_CONCEPTS.items()):
            print(f"\n{concept.upper()}:")
            for term in terms:
                print(f"  - {term}")
        print("\n" + "=" * 70)
        print("\nNote: You can use --use-llm for dynamic concept extraction")
        print("or train a custom model with your own economic domain knowledge")
        return
    
    # Create training template
    if args.create_training_template:
        print("\nNote: Training template creation is currently commented out.")
        print("Uncomment the EconomicConceptTrainer class and related functions")
        print("in the source code to enable custom model training.")
        print("\nFor now, use --use-llm for zero-shot classification with pre-trained models")
        return
    
    # Check for query
    if not args.query:
        parser.print_help()
        print("\nPlease provide a research question or use --list-concepts")
        return
    
    query = ' '.join(args.query)
    
    # Warn about LLM usage
    if args.use_llm and not LLM_AVAILABLE:
        print("Warning: --use-llm specified but transformers library not installed")
        print("Install with: pip install transformers torch")
        print("Falling back to rule-based extraction\n")
    
    assistant = FREDResearchAssistant(args.project_dir, use_llm=args.use_llm)
    
    # Recommend only mode
    if args.recommend_only:
        recommendation = assistant.recommend_series(query)
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print(f"\nQuery: {query}")
        print(f"Category: {recommendation['category']}")
        print(f"Method: {recommendation['extraction_method']}")
        print(f"\nRecommended search terms ({recommendation['num_terms']}):")
        for term in recommendation['search_terms']:
            print(f"  - {term}")
        print()
        return
    
    # Run full pipeline
    assistant.run_pipeline(
        query,
        limit=args.limit,
        skip_fetch=args.skip_fetch,
        skip_parse=args.skip_parse,
        skip_embeddings=args.skip_embeddings,
        skip_monolith=args.skip_monolith
    )


if __name__ == '__main__':
    main()
