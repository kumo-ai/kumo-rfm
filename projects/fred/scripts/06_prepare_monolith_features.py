#!/usr/bin/env python3
"""
Prepare FRED data for ByteDance Monolith recommendation algorithm.
Transforms series metadata into feature format and exports as TFRecord.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json


class MonolithFeaturePreparation:
    """Prepare features for ByteDance Monolith algorithm."""
    
    def __init__(self):
        self.feature_config = {}
        self.vocabulary = {}
        
    def create_categorical_features(self, series_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create comprehensive categorical features for Monolith.
        
        Returns dictionary of feature tables.
        """
        print("Creating categorical features...")
        
        features = {}
        
        # Series ID features (item features)
        features['series_id'] = pd.DataFrame({
            'series_id': series_df['series_id'],
            'series_id_hash': series_df['series_id'].apply(lambda x: hash(x) % (10**9))
        })
        
        # Category features
        categories = series_df['category'].unique()
        self.vocabulary['category'] = {cat: idx for idx, cat in enumerate(categories)}
        features['category'] = pd.DataFrame({
            'series_id': series_df['series_id'],
            'category': series_df['category'],
            'category_id': series_df['category'].map(self.vocabulary['category'])
        })
        
        # Frequency features
        frequencies = series_df['frequency'].unique()
        self.vocabulary['frequency'] = {freq: idx for idx, freq in enumerate(frequencies)}
        features['frequency'] = pd.DataFrame({
            'series_id': series_df['series_id'],
            'frequency': series_df['frequency'],
            'frequency_id': series_df['frequency'].map(self.vocabulary['frequency'])
        })
        
        # Seasonal adjustment features
        if 'seasonal_adjustment' in series_df.columns:
            seasonal_adj = series_df['seasonal_adjustment'].fillna('Not Available').unique()
            self.vocabulary['seasonal_adjustment'] = {sa: idx for idx, sa in enumerate(seasonal_adj)}
            features['seasonal_adjustment'] = pd.DataFrame({
                'series_id': series_df['series_id'],
                'seasonal_adjustment': series_df['seasonal_adjustment'].fillna('Not Available'),
                'seasonal_adjustment_id': series_df['seasonal_adjustment'].fillna('Not Available').map(self.vocabulary['seasonal_adjustment'])
            })
        
        # Units features (data type)
        if 'units' in series_df.columns:
            units = series_df['units'].fillna('Unknown').unique()
            # Limit vocabulary size for units (can be very large)
            top_units = series_df['units'].fillna('Unknown').value_counts().head(100).index
            units_limited = [u if u in top_units else 'Other' for u in series_df['units'].fillna('Unknown')]
            unique_units = list(set(units_limited))
            self.vocabulary['units'] = {unit: idx for idx, unit in enumerate(unique_units)}
            features['units'] = pd.DataFrame({
                'series_id': series_df['series_id'],
                'units': units_limited,
                'units_id': [self.vocabulary['units'][u] for u in units_limited]
            })
        
        # Create combined categorical features
        if 'category' in series_df.columns and 'frequency' in series_df.columns:
            # Category x Frequency combinations
            cat_freq_combo = series_df['category'] + '|' + series_df['frequency']
            cat_freq_unique = cat_freq_combo.unique()
            self.vocabulary['category_frequency'] = {cf: idx for idx, cf in enumerate(cat_freq_unique)}
            features['category_frequency'] = pd.DataFrame({
                'series_id': series_df['series_id'],
                'category_frequency': cat_freq_combo,
                'category_frequency_id': cat_freq_combo.map(self.vocabulary['category_frequency'])
            })
        
        # Extract keywords from title as categorical features
        if 'title' in series_df.columns:
            # Common economic keywords
            keywords = ['rate', 'index', 'price', 'employment', 'gdp', 'inflation', 'consumer', 
                       'producer', 'housing', 'retail', 'manufacturing', 'trade', 'debt', 
                       'stock', 'bond', 'interest', 'bank', 'credit', 'wage', 'income']
            
            for keyword in keywords:
                col_name = f'has_{keyword}'
                features[col_name] = pd.DataFrame({
                    'series_id': series_df['series_id'],
                    col_name: series_df['title'].str.lower().str.contains(keyword, na=False).astype(int)
                })
        
        print(f"Created {len(features)} categorical feature tables")
        return features
    
    def create_numerical_features(self, series_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive numerical features for Monolith."""
        print("Creating numerical features...")
        
        # Basic popularity features
        pop_mean = series_df['popularity'].mean()
        pop_std = series_df['popularity'].std()
        pop_median = series_df['popularity'].median()
        
        numerical_features = pd.DataFrame({
            'series_id': series_df['series_id'],
            
            # Raw popularity
            'popularity': series_df['popularity'],
            'popularity_normalized': (series_df['popularity'] - pop_mean) / pop_std,
            'popularity_log': np.log1p(series_df['popularity']),
            'popularity_rank': series_df['popularity'].rank(pct=True),
            'popularity_vs_median': series_df['popularity'] / pop_median,
            
            # Binned popularity (for categorical treatment)
            'popularity_bin': pd.qcut(series_df['popularity'], q=10, labels=False, duplicates='drop'),
            'is_high_popularity': (series_df['popularity'] > series_df['popularity'].quantile(0.75)).astype(int),
            'is_low_popularity': (series_df['popularity'] < series_df['popularity'].quantile(0.25)).astype(int),
            
            # Notes features
            'notes_length': series_df['notes_length'].fillna(0),
            'notes_length_log': np.log1p(series_df['notes_length'].fillna(0)),
            'has_notes': series_df['has_notes'].astype(int),
            'notes_length_normalized': (series_df['notes_length'].fillna(0) - series_df['notes_length'].fillna(0).mean()) / (series_df['notes_length'].fillna(0).std() + 1e-6),
        })
        
        # Add title length features (proxy for complexity/specificity)
        if 'title' in series_df.columns:
            numerical_features['title_length'] = series_df['title'].str.len()
            numerical_features['title_word_count'] = series_df['title'].str.split().str.len()
            numerical_features['title_length_log'] = np.log1p(numerical_features['title_length'])
        
        # Category-based features
        if 'category' in series_df.columns:
            # Popularity within category
            cat_pop_mean = series_df.groupby('category')['popularity'].transform('mean')
            cat_pop_std = series_df.groupby('category')['popularity'].transform('std')
            numerical_features['popularity_vs_category_mean'] = series_df['popularity'] / (cat_pop_mean + 1e-6)
            numerical_features['popularity_z_in_category'] = (series_df['popularity'] - cat_pop_mean) / (cat_pop_std + 1e-6)
            
            # Category size (how many series in this category)
            cat_counts = series_df['category'].value_counts()
            numerical_features['category_size'] = series_df['category'].map(cat_counts)
            numerical_features['category_size_log'] = np.log1p(numerical_features['category_size'])
        
        # Frequency-based features
        if 'frequency' in series_df.columns:
            freq_pop_mean = series_df.groupby('frequency')['popularity'].transform('mean')
            numerical_features['popularity_vs_frequency_mean'] = series_df['popularity'] / (freq_pop_mean + 1e-6)
            
            # Frequency type indicators
            numerical_features['is_daily'] = (series_df['frequency'] == 'Daily').astype(int)
            numerical_features['is_monthly'] = (series_df['frequency'] == 'Monthly').astype(int)
            numerical_features['is_quarterly'] = (series_df['frequency'] == 'Quarterly').astype(int)
            numerical_features['is_annual'] = (series_df['frequency'] == 'Annual').astype(int)
        
        # Seasonal adjustment features
        if 'seasonal_adjustment' in series_df.columns:
            numerical_features['is_seasonally_adjusted'] = (series_df['seasonal_adjustment'] == 'Seasonally Adjusted').astype(int)
        
        # Units features (can indicate data type)
        if 'units' in series_df.columns:
            numerical_features['is_percent'] = series_df['units'].str.contains('Percent|percent|%', na=False).astype(int)
            numerical_features['is_index'] = series_df['units'].str.contains('Index|index', na=False).astype(int)
            numerical_features['is_dollars'] = series_df['units'].str.contains('Dollar|dollar|\$', na=False).astype(int)
            numerical_features['is_rate'] = series_df['units'].str.contains('Rate|rate', na=False).astype(int)
        
        # Cross-feature interactions
        numerical_features['pop_times_notes'] = numerical_features['popularity_log'] * numerical_features['notes_length_log']
        if 'title_length' in numerical_features.columns:
            numerical_features['pop_times_title'] = numerical_features['popularity_log'] * numerical_features['title_length_log']
        
        return numerical_features
    
    def create_user_item_interactions(self, series_df: pd.DataFrame, n_synthetic: int = 1000) -> pd.DataFrame:
        """
        Create synthetic user-item interactions for demonstration.
        
        In production, this would be real user query/view logs.
        """
        print(f"Creating {n_synthetic} synthetic user interactions...")
        
        # Simulate users with different interests
        interactions = []
        
        for user_id in range(n_synthetic):
            # Each user queries 1-10 series
            n_queries = np.random.randint(1, 11)
            
            # Users tend to query popular series
            weights = series_df['popularity'].fillna(0).values ** 2
            weights = weights / weights.sum()
            
            sampled_series = np.random.choice(
                series_df['series_id'].values,
                size=n_queries,
                replace=False,
                p=weights
            )
            
            for idx, series_id in enumerate(sampled_series):
                series_data = series_df[series_df['series_id'] == series_id].iloc[0]
                
                interactions.append({
                    'user_id': f'user_{user_id}',
                    'series_id': series_id,
                    'category': series_data['category'],
                    'frequency': series_data['frequency'],
                    'popularity': series_data['popularity'],
                    'timestamp': 1609459200 + (user_id * 86400) + (idx * 3600),  # Synthetic timestamps
                    'label': 1  # All positive interactions for demo
                })
        
        interactions_df = pd.DataFrame(interactions)
        print(f"Created {len(interactions_df)} interactions for {n_synthetic} users")
        
        return interactions_df
    
    def create_sequence_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sequence features (user query history).
        
        Monolith can use sequential patterns for recommendations.
        """
        print("Creating sequence features...")
        
        # Group by user and create sequences
        sequences = []
        
        for user_id, group in interactions_df.groupby('user_id'):
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Create sequence of queried series
            series_sequence = group['series_id'].tolist()
            category_sequence = group['category'].tolist()
            
            sequences.append({
                'user_id': user_id,
                'series_sequence': series_sequence,
                'series_sequence_length': len(series_sequence),
                'category_sequence': category_sequence,
                'latest_category': category_sequence[-1] if category_sequence else None
            })
        
        sequences_df = pd.DataFrame(sequences)
        print(f"Created sequences for {len(sequences_df)} users")
        
        return sequences_df
    
    def prepare_monolith_format(self, 
                                series_df: pd.DataFrame,
                                interactions_df: pd.DataFrame,
                                embeddings: np.ndarray = None) -> Dict[str, Any]:
        """
        Prepare complete feature set in Monolith format.
        
        Returns dictionary with all feature tables and metadata.
        """
        print("\n" + "="*60)
        print("PREPARING MONOLITH FEATURES")
        print("="*60)
        
        monolith_data = {}
        
        # Item features (series metadata)
        categorical_features = self.create_categorical_features(series_df)
        numerical_features = self.create_numerical_features(series_df)
        
        # Combine item features
        item_features = series_df[['series_id']].copy()
        for feat_name, feat_df in categorical_features.items():
            item_features = item_features.merge(feat_df, on='series_id', how='left')
        item_features = item_features.merge(numerical_features, on='series_id', how='left')
        
        # Add embeddings if available
        if embeddings is not None:
            print("Adding dense embeddings to item features...")
            embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
            embedding_df['series_id'] = series_df['series_id'].values
            item_features = item_features.merge(embedding_df, on='series_id', how='left')
        
        monolith_data['item_features'] = item_features
        
        # User interaction features
        sequence_features = self.create_sequence_features(interactions_df)
        monolith_data['user_features'] = sequence_features
        
        # Training samples (user-item pairs with labels)
        monolith_data['training_samples'] = interactions_df
        
        # Feature config (for Monolith model configuration)
        monolith_data['feature_config'] = {
            'categorical_features': list(categorical_features.keys()),
            'numerical_features': list(numerical_features.columns),
            'sequence_features': ['series_sequence', 'category_sequence'],
            'embedding_dim': embeddings.shape[1] if embeddings is not None else 0,
            'vocabularies': self.vocabulary
        }
        
        print("\nMonolith data preparation complete!")
        print(f"  - Item features: {len(item_features)} series")
        print(f"  - User features: {len(sequence_features)} users")
        print(f"  - Training samples: {len(interactions_df)} interactions")
        
        return monolith_data
    
    def export_to_tfrecord(self, monolith_data: Dict[str, Any], output_dir: str):
        """
        Export prepared data to TFRecord format for Monolith training.
        """
        try:
            import tensorflow as tf
        except ImportError:
            print("TensorFlow not installed. Cannot export TFRecord.")
            print("Install with: pip install tensorflow")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\nExporting to TFRecord format at {output_dir}...")
        
        # Export training samples
        train_file = output_path / 'train.tfrecord'
        with tf.io.TFRecordWriter(str(train_file)) as writer:
            for _, row in monolith_data['training_samples'].iterrows():
                # Get item features for this series
                item_feat = monolith_data['item_features'][
                    monolith_data['item_features']['series_id'] == row['series_id']
                ].iloc[0]
                
                # Create feature dict
                feature = {
                    'user_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['user_id'].encode()])),
                    'series_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['series_id'].encode()])),
                    'category_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(item_feat.get('category_id', 0))])),
                    'frequency_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(item_feat.get('frequency_id', 0))])),
                    'popularity': tf.train.Feature(float_list=tf.train.FloatList(value=[float(row['popularity'])])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row['label'])]))
                }
                
                # Add embeddings if present
                emb_cols = [col for col in item_feat.index if col.startswith('emb_')]
                if emb_cols:
                    emb_values = [float(item_feat[col]) for col in emb_cols]
                    feature['embedding'] = tf.train.Feature(float_list=tf.train.FloatList(value=emb_values))
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        
        print(f"Exported training data to {train_file}")
        
        # Export feature config as JSON
        config_file = output_path / 'feature_config.json'
        with open(config_file, 'w') as f:
            # Convert to serializable format
            config = monolith_data['feature_config'].copy()
            config['vocabularies'] = {k: {str(k2): int(v2) for k2, v2 in v.items()} 
                                     for k, v in config['vocabularies'].items()}
            json.dump(config, f, indent=2)
        
        print(f"Exported feature config to {config_file}")
    
    def export_to_parquet(self, monolith_data: Dict[str, Any], output_dir: str):
        """Export to Parquet format (alternative to TFRecord)."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\nExporting to Parquet format at {output_dir}...")
        
        # Export each table
        monolith_data['item_features'].to_parquet(output_path / 'item_features.parquet', index=False)
        monolith_data['user_features'].to_parquet(output_path / 'user_features.parquet', index=False)
        monolith_data['training_samples'].to_parquet(output_path / 'training_samples.parquet', index=False)
        
        # Export config
        config_file = output_path / 'feature_config.json'
        with open(config_file, 'w') as f:
            config = monolith_data['feature_config'].copy()
            config['vocabularies'] = {k: {str(k2): int(v2) for k2, v2 in v.items()} 
                                     for k, v in config['vocabularies'].items()}
            json.dump(config, f, indent=2)
        
        print(f"Exported all tables to Parquet format")


def main():
    parser = argparse.ArgumentParser(description='Prepare FRED data for ByteDance Monolith')
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Path to FRED series data')
    parser.add_argument('--embeddings', type=str,
                       help='Path to embeddings .npy file (optional)')
    parser.add_argument('--output', type=str, default='data/monolith',
                       help='Output directory')
    parser.add_argument('--n-interactions', type=int, default=1000,
                       help='Number of synthetic interactions to generate')
    parser.add_argument('--format', type=str, default='parquet',
                       choices=['tfrecord', 'parquet', 'both'],
                       help='Output format')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BYTEDANCE MONOLITH FEATURE PREPARATION")
    print("="*60)
    
    # Load data
    print(f"\nLoading FRED data from {args.data}...")
    series_df = pd.read_parquet(args.data)
    print(f"Loaded {len(series_df)} series")
    
    # Load embeddings if provided
    embeddings = None
    if args.embeddings and Path(args.embeddings).exists():
        print(f"Loading embeddings from {args.embeddings}...")
        embeddings = np.load(args.embeddings)
        print(f"Loaded embeddings with shape {embeddings.shape}")
    
    # Initialize preparation
    prep = MonolithFeaturePreparation()
    
    # Create synthetic interactions
    interactions_df = prep.create_user_item_interactions(series_df, n_synthetic=args.n_interactions)
    
    # Prepare Monolith format
    monolith_data = prep.prepare_monolith_format(series_df, interactions_df, embeddings)
    
    # Export
    if args.format in ['parquet', 'both']:
        prep.export_to_parquet(monolith_data, args.output)
    
    if args.format in ['tfrecord', 'both']:
        prep.export_to_tfrecord(monolith_data, args.output)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"\nData exported to: {args.output}")
    print("\nNext steps:")
    print("1. Review feature_config.json for model configuration")
    print("2. Use training_samples for model training")
    print("3. Use item_features for inference/serving")


if __name__ == '__main__':
    main()
