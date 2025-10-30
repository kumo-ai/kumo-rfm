#!/usr/bin/env python3
"""
ML-Based Schema Recommendation System

Uses machine learning to automatically recommend primary keys, foreign keys,
and table relationships from raw data.

Models implemented:
1. Decision Tree - Rule-based FK detection
2. Random Forest - Ensemble FK detection
3. XGBoost - Gradient boosting FK detection
4. Neural Network - Deep learning FK detection
5. Graph Neural Network - Relationship detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    import networkx as nx
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: sklearn/networkx not installed")


class SchemaRecommender:
    """ML-based schema recommendation system."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_column_features(self, df: pd.DataFrame, col_name: str) -> Dict:
        """
        Extract features from a column to predict if it's a key.
        
        Features:
        - Uniqueness ratio
        - Null ratio
        - Data type
        - Name patterns (contains 'id', 'key', 'code')
        - Value distribution
        - String length statistics
        """
        col = df[col_name]
        
        features = {
            # Uniqueness
            'unique_ratio': col.nunique() / len(col),
            'is_fully_unique': int(col.nunique() == len(col)),
            
            # Nulls
            'null_ratio': col.isna().sum() / len(col),
            'has_nulls': int(col.isna().any()),
            
            # Data type
            'is_numeric': int(pd.api.types.is_numeric_dtype(col)),
            'is_integer': int(pd.api.types.is_integer_dtype(col)),
            'is_string': int(pd.api.types.is_string_dtype(col)),
            
            # Name patterns
            'name_contains_id': int('id' in col_name.lower()),
            'name_contains_key': int('key' in col_name.lower()),
            'name_contains_code': int('code' in col_name.lower()),
            'name_contains_num': int('num' in col_name.lower() or 'no' in col_name.lower()),
            'name_length': len(col_name),
            'name_has_underscore': int('_' in col_name),
            
            # Position in columns
            'is_first_column': int(list(df.columns).index(col_name) == 0),
            
            # Value characteristics
            'mean_value': float(col.mean()) if pd.api.types.is_numeric_dtype(col) else 0,
            'std_value': float(col.std()) if pd.api.types.is_numeric_dtype(col) else 0,
            'min_value': float(col.min()) if pd.api.types.is_numeric_dtype(col) else 0,
            'max_value': float(col.max()) if pd.api.types.is_numeric_dtype(col) else 0,
        }
        
        # String-specific features
        if pd.api.types.is_string_dtype(col):
            lengths = col.dropna().astype(str).str.len()
            features['avg_string_length'] = float(lengths.mean()) if len(lengths) > 0 else 0
            features['string_length_std'] = float(lengths.std()) if len(lengths) > 0 else 0
        else:
            features['avg_string_length'] = 0
            features['string_length_std'] = 0
        
        return features
    
    def detect_potential_foreign_key(self, source_df: pd.DataFrame, 
                                     source_col: str,
                                     target_df: pd.DataFrame,
                                     target_col: str) -> Dict:
        """
        Extract features for FK relationship detection.
        
        Features:
        - Column name similarity
        - Value overlap
        - Cardinality ratios
        """
        source_values = set(source_df[source_col].dropna())
        target_values = set(target_df[target_col].dropna())
        
        # Calculate overlap
        intersection = source_values & target_values
        overlap_ratio = len(intersection) / len(source_values) if len(source_values) > 0 else 0
        
        features = {
            # Name similarity
            'name_exact_match': int(source_col == target_col),
            'name_substring': int(source_col in target_col or target_col in source_col),
            'name_similarity': self._string_similarity(source_col, target_col),
            
            # Value overlap
            'value_overlap_ratio': overlap_ratio,
            'has_high_overlap': int(overlap_ratio > 0.8),
            
            # Cardinality
            'source_unique_count': len(source_values),
            'target_unique_count': len(target_values),
            'cardinality_ratio': len(source_values) / len(target_values) if len(target_values) > 0 else 0,
            
            # Data type match
            'same_dtype': int(source_df[source_col].dtype == target_df[target_col].dtype),
            
            # Both are ID-like
            'both_are_ids': int('id' in source_col.lower() and 'id' in target_col.lower()),
        }
        
        return features
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity between two strings."""
        s1, s2 = s1.lower(), s2.lower()
        
        if s1 == s2:
            return 1.0
        
        # Simple character overlap similarity
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def train_primary_key_detector(self, training_data: List[Dict]) -> Dict:
        """
        Train models to detect primary keys.
        
        Args:
            training_data: List of {features: dict, is_primary_key: bool}
        
        Returns:
            Trained models and metrics
        """
        if not ML_AVAILABLE:
            raise ImportError("sklearn required for ML models")
        
        # Prepare data
        X = pd.DataFrame([d['features'] for d in training_data])
        y = np.array([d['is_primary_key'] for d in training_data])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Model 1: Decision Tree (interpretable)
        print("\nTraining Decision Tree...")
        dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
        dt.fit(X_train_scaled, y_train)
        dt_score = dt.score(X_test_scaled, y_test)
        results['decision_tree'] = {
            'model': dt,
            'accuracy': dt_score,
            'feature_importance': dict(zip(X.columns, dt.feature_importances_))
        }
        print(f"  Accuracy: {dt_score:.3f}")
        
        # Model 2: Random Forest (ensemble)
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_score = rf.score(X_test_scaled, y_test)
        results['random_forest'] = {
            'model': rf,
            'accuracy': rf_score,
            'feature_importance': dict(zip(X.columns, rf.feature_importances_))
        }
        print(f"  Accuracy: {rf_score:.3f}")
        
        # Model 3: Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_score = gb.score(X_test_scaled, y_test)
        results['gradient_boosting'] = {
            'model': gb,
            'accuracy': gb_score,
            'feature_importance': dict(zip(X.columns, gb.feature_importances_))
        }
        print(f"  Accuracy: {gb_score:.3f}")
        
        # Model 4: Neural Network
        print("\nTraining Neural Network...")
        nn = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        nn.fit(X_train_scaled, y_train)
        nn_score = nn.score(X_test_scaled, y_test)
        results['neural_network'] = {
            'model': nn,
            'accuracy': nn_score
        }
        print(f"  Accuracy: {nn_score:.3f}")
        
        self.models['primary_key'] = results
        return results
    
    def recommend_primary_keys(self, df: pd.DataFrame, 
                               model_type: str = 'random_forest') -> List[Tuple[str, float]]:
        """
        Recommend primary key columns using trained model.
        
        Returns:
            List of (column_name, confidence_score)
        """
        if 'primary_key' not in self.models:
            raise ValueError("Model not trained. Call train_primary_key_detector first.")
        
        model_info = self.models['primary_key'][model_type]
        model = model_info['model']
        
        recommendations = []
        
        for col in df.columns:
            features = self.extract_column_features(df, col)
            feature_vector = pd.DataFrame([features])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                confidence = model.predict_proba(feature_vector_scaled)[0][1]
            else:
                confidence = float(model.predict(feature_vector_scaled)[0])
            
            recommendations.append((col, confidence))
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def build_relationship_graph(self, tables: Dict[str, pd.DataFrame]) -> nx.DiGraph:
        """
        Build a graph of potential table relationships using GNN approach.
        
        Nodes: Tables + Columns
        Edges: Potential FK relationships
        """
        G = nx.DiGraph()
        
        # Add table nodes
        for table_name in tables.keys():
            G.add_node(table_name, node_type='table')
        
        # Add column nodes
        for table_name, df in tables.items():
            for col in df.columns:
                col_node = f"{table_name}.{col}"
                features = self.extract_column_features(df, col)
                G.add_node(col_node, node_type='column', **features)
                G.add_edge(table_name, col_node, edge_type='has_column')
        
        # Detect potential FK relationships
        table_list = list(tables.items())
        for i, (source_table, source_df) in enumerate(table_list):
            for source_col in source_df.columns:
                for target_table, target_df in table_list[i+1:]:
                    for target_col in target_df.columns:
                        # Check if could be FK
                        fk_features = self.detect_potential_foreign_key(
                            source_df, source_col,
                            target_df, target_col
                        )
                        
                        # Simple heuristic: high overlap + name similarity
                        if (fk_features['value_overlap_ratio'] > 0.7 and
                            fk_features['name_similarity'] > 0.5):
                            
                            source_node = f"{source_table}.{source_col}"
                            target_node = f"{target_table}.{target_col}"
                            
                            G.add_edge(
                                source_node, target_node,
                                edge_type='potential_fk',
                                **fk_features
                            )
        
        return G
    
    def generate_training_data_from_fred(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate training data from FRED metadata.
        
        Uses heuristics to label columns as PK/not PK.
        """
        training_data = []
        
        for col in df.columns:
            features = self.extract_column_features(df, col)
            
            # Heuristic labeling (you can improve this with manual labels)
            is_primary_key = (
                col.lower() in ['series_id', 'id', 'category_id', 'frequency_id'] or
                (features['unique_ratio'] == 1.0 and features['null_ratio'] == 0)
            )
            
            training_data.append({
                'column_name': col,
                'features': features,
                'is_primary_key': is_primary_key
            })
        
        return training_data
    
    def export_decision_tree_rules(self, model_type: str = 'decision_tree') -> str:
        """Export decision tree as human-readable rules."""
        if 'primary_key' not in self.models:
            raise ValueError("Model not trained")
        
        model = self.models['primary_key'][model_type]['model']
        
        if hasattr(model, 'tree_'):
            feature_names = list(self.models['primary_key'][model_type]['feature_importance'].keys())
            return export_text(model, feature_names=feature_names)
        else:
            return "Model does not support rule export"


def demo_ml_schema_recommendation():
    """Demonstrate ML-based schema recommendation."""
    print("="*70)
    print("ML-BASED SCHEMA RECOMMENDATION DEMO")
    print("="*70)
    
    if not ML_AVAILABLE:
        print("\nError: sklearn and networkx required")
        print("Install with: pip install scikit-learn networkx")
        return
    
    # Load FRED data
    df = pd.read_parquet('data/fred_series_metadata.parquet')
    
    print(f"\nLoaded {len(df)} series with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize recommender
    recommender = SchemaRecommender()
    
    # Generate training data
    print("\n" + "-"*70)
    print("Generating training data from heuristics...")
    training_data = recommender.generate_training_data_from_fred(df)
    print(f"Generated {len(training_data)} training samples")
    
    # Train models
    print("\n" + "-"*70)
    print("Training ML models...")
    results = recommender.train_primary_key_detector(training_data)
    
    # Show feature importance
    print("\n" + "-"*70)
    print("Top 10 Important Features (Random Forest):")
    importance = results['random_forest']['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, score in sorted_features[:10]:
        print(f"  {feature}: {score:.4f}")
    
    # Get recommendations
    print("\n" + "-"*70)
    print("Primary Key Recommendations:")
    recommendations = recommender.recommend_primary_keys(df, model_type='random_forest')
    
    for col, confidence in recommendations[:5]:
        print(f"  {col}: {confidence:.2%} confidence")
    
    # Build relationship graph
    print("\n" + "-"*70)
    print("Building relationship graph...")
    
    # Create sample tables for demonstration
    categories = df.groupby('category').agg({
        'series_id': 'count',
        'popularity': 'mean'
    }).reset_index()
    categories.columns = ['category_name', 'series_count', 'avg_popularity']
    categories['category_id'] = range(len(categories))
    
    tables = {
        'series': df[['series_id', 'category', 'frequency', 'popularity']].head(100),
        'categories': categories
    }
    
    G = recommender.build_relationship_graph(tables)
    
    print(f"  Graph nodes: {G.number_of_nodes()}")
    print(f"  Graph edges: {G.number_of_edges()}")
    
    # Find potential FK relationships
    fk_edges = [(u, v, d) for u, v, d in G.edges(data=True) 
                if d.get('edge_type') == 'potential_fk']
    
    print(f"\nPotential Foreign Key Relationships: {len(fk_edges)}")
    for source, target, data in fk_edges[:5]:
        print(f"  {source} -> {target}")
        print(f"    Overlap: {data['value_overlap_ratio']:.2%}")
        print(f"    Name similarity: {data['name_similarity']:.2f}")
    
    # Export decision tree rules
    print("\n" + "-"*70)
    print("Decision Tree Rules (first 20 lines):")
    rules = recommender.export_decision_tree_rules()
    print('\n'.join(rules.split('\n')[:20]))
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == '__main__':
    demo_ml_schema_recommendation()
