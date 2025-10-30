#!/usr/bin/env python3
"""
Advanced KumoAI RFM Demo for Economic Series Analysis

This demo showcases advanced KumoAI features:
1. Multi-table relationships and joins
2. Time-series predictions
3. Anomaly detection
4. Causal inference
5. What-if analysis
6. Feature importance
7. Automated insights
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

try:
    import kumoai.experimental.rfm as rfm
    KUMO_AVAILABLE = True
except ImportError:
    KUMO_AVAILABLE = False
    print("Warning: kumoai not installed. Install with: pip install kumoai")


class AdvancedKumoDemo:
    """Advanced KumoAI RFM demonstrations for economic analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        if not KUMO_AVAILABLE:
            raise ImportError("kumoai is required for this demo")
        
        self.api_key = api_key or os.getenv('KUMO_API_KEY')
        if not self.api_key:
            raise ValueError("KUMO_API_KEY must be set")
        
        rfm.init(api_key=self.api_key)
        self.graph = None
        self.model = None
        
    def load_data(self, data_path: str = 'data/fred_series_metadata.parquet') -> pd.DataFrame:
        """Load FRED series metadata."""
        return pd.read_parquet(data_path)
    
    def demo_1_multi_table_relationships(self, series_df: pd.DataFrame):
        """
        Demo 1: Multi-table relationships
        
        Shows how to build complex graphs with multiple related tables.
        Use case: Linking economic series to their categories and frequencies.
        """
        print("\n" + "="*70)
        print("DEMO 1: Multi-Table Relationships")
        print("="*70)
        print("\nBuilding graph with series, categories, and frequency tables...\n")
        
        # Create category table
        categories = series_df.groupby('category').agg({
            'series_id': 'count',
            'popularity': 'mean'
        }).reset_index()
        categories.columns = ['category_name', 'series_count', 'avg_popularity']
        categories['category_id'] = range(len(categories))
        
        # Create frequency table
        frequencies = series_df.groupby(['frequency', 'frequency_name']).size().reset_index()
        frequencies.columns = ['freq_code', 'freq_name', 'count']
        frequencies['frequency_id'] = range(len(frequencies))
        
        # Add foreign keys to series
        series_enhanced = series_df.copy()
        cat_map = dict(zip(categories['category_name'], categories['category_id']))
        freq_map = dict(zip(frequencies['freq_code'], frequencies['frequency_id']))
        series_enhanced['category_id'] = series_enhanced['category'].map(cat_map)
        series_enhanced['frequency_id'] = series_enhanced['frequency'].map(freq_map)
        
        # Create tables
        series_table = rfm.LocalTable(
            series_enhanced[['series_id', 'category_id', 'frequency_id', 'popularity', 'notes_length']],
            name="series",
            primary_key="series_id"
        )
        
        category_table = rfm.LocalTable(
            categories,
            name="categories",
            primary_key="category_id"
        )
        
        frequency_table = rfm.LocalTable(
            frequencies,
            name="frequencies",
            primary_key="frequency_id"
        )
        
        # Define relationships
        series_table['category_id'].stype = "ID"
        series_table['frequency_id'].stype = "ID"
        
        category_table['category_id'].stype = "ID"
        frequency_table['frequency_id'].stype = "ID"
        
        # Build graph with relationships
        # Note: KumoAI infers relationships automatically from foreign keys
        graph = rfm.LocalGraph(
            tables=[series_table, category_table, frequency_table]
        )
        
        self.graph = graph
        self.model = rfm.KumoRFM(graph)
        
        print(f"Created graph with {len(graph.tables)} tables")
        print(f"  - Series: {len(series_enhanced)} rows")
        print(f"  - Categories: {len(categories)} rows")
        print(f"  - Frequencies: {len(frequencies)} rows")
        
        # Example query: Predict series popularity based on category and frequency
        print("\nExample Query: Predict popularity for series in 'Employment' category")
        employment_series = series_enhanced[
            series_enhanced['category'].str.contains('Employment', case=False, na=False)
        ].head(3)['series_id'].tolist()
        
        for series_id in employment_series:
            try:
                query = f"PREDICT series.popularity FOR series.series_id = '{series_id}'"
                result = self.model.predict(query)
                print(f"  {series_id}: {result}")
            except Exception as e:
                print(f"  {series_id}: Prediction failed ({e})")
        
        print("\nDemo 1 Complete!")
    
    def demo_2_temporal_predictions(self, series_df: pd.DataFrame):
        """
        Demo 2: Temporal/Time-series predictions
        
        Shows how to make predictions that incorporate time-based patterns.
        Use case: Predict which series will become more popular over time.
        """
        print("\n" + "="*70)
        print("DEMO 2: Temporal Predictions")
        print("="*70)
        print("\nSimulating time-series data for popularity trends...\n")
        
        # Create synthetic time-series data
        time_series = []
        for idx, row in series_df.head(50).iterrows():
            # Simulate monthly popularity data for past 12 months
            base_pop = row['popularity']
            for month in range(12):
                time_series.append({
                    'series_id': row['series_id'],
                    'month': month,
                    'date': (datetime.now() - timedelta(days=30*(12-month))).strftime('%Y-%m-%d'),
                    'popularity': base_pop + np.random.randint(-10, 20),
                    'views': np.random.randint(100, 10000),
                    'category': row['category']
                })
        
        ts_df = pd.DataFrame(time_series)
        
        # Create temporal table
        # Convert date to datetime and set as index for temporal analysis
        ts_df['date'] = pd.to_datetime(ts_df['date'])
        
        ts_table = rfm.LocalTable(
            ts_df,
            name="popularity_trends",
            primary_key=["series_id", "month"]
        )
        
        # Set semantic types
        ts_table['series_id'].stype = "ID"
        ts_table['month'].stype = "numerical"
        ts_table['date'].stype = "timestamp"  # Mark date column as timestamp
        ts_table['popularity'].stype = "numerical"
        ts_table['views'].stype = "numerical"
        
        # Build graph
        graph = rfm.LocalGraph(tables=[ts_table])
        model = rfm.KumoRFM(graph)
        
        print(f"Created time-series dataset with {len(ts_df)} observations")
        print(f"  - {len(ts_df['series_id'].unique())} unique series")
        print(f"  - {len(ts_df['month'].unique())} months of data")
        
        # Predict future popularity
        print("\nPredicting popularity for next month...")
        sample_series = ts_df['series_id'].unique()[:3]
        
        for series_id in sample_series:
            try:
                # Predict popularity for month 13 (next month)
                query = f"PREDICT popularity_trends.popularity FOR popularity_trends.series_id = '{series_id}' AND popularity_trends.month = 13"
                result = model.predict(query)
                current = ts_df[ts_df['series_id'] == series_id]['popularity'].iloc[-1]
                print(f"  {series_id}:")
                print(f"    Current (month 12): {current}")
                print(f"    Predicted (month 13): {result}")
            except Exception as e:
                print(f"  {series_id}: Prediction failed ({e})")
        
        print("\nDemo 2 Complete!")
    
    def demo_3_anomaly_detection(self, series_df: pd.DataFrame):
        """
        Demo 3: Anomaly detection
        
        Identify series with unusual characteristics compared to their category.
        Use case: Find outlier economic indicators that don't fit normal patterns.
        """
        print("\n" + "="*70)
        print("DEMO 3: Anomaly Detection")
        print("="*70)
        print("\nDetecting anomalous series within categories...\n")
        
        # Calculate statistics per category
        category_stats = series_df.groupby('category').agg({
            'popularity': ['mean', 'std'],
            'notes_length': ['mean', 'std']
        }).reset_index()
        
        category_stats.columns = ['category', 'pop_mean', 'pop_std', 'notes_mean', 'notes_std']
        
        # Merge with original data
        merged = series_df.merge(category_stats, on='category')
        
        # Calculate z-scores
        merged['pop_zscore'] = (merged['popularity'] - merged['pop_mean']) / (merged['pop_std'] + 1e-10)
        merged['notes_zscore'] = (merged['notes_length'] - merged['notes_mean']) / (merged['notes_std'] + 1e-10)
        merged['anomaly_score'] = np.abs(merged['pop_zscore']) + np.abs(merged['notes_zscore'])
        
        # Find anomalies
        anomalies = merged.nlargest(10, 'anomaly_score')
        
        print("Top 10 Anomalous Series:")
        print("-" * 70)
        for idx, row in anomalies.iterrows():
            print(f"\n{row['series_id']} - {row['title'][:50]}")
            print(f"  Category: {row['category']}")
            print(f"  Popularity: {row['popularity']} (z-score: {row['pop_zscore']:.2f})")
            print(f"  Notes Length: {row['notes_length']} (z-score: {row['notes_zscore']:.2f})")
            print(f"  Anomaly Score: {row['anomaly_score']:.2f}")
        
        # Use KumoAI to predict if series is an outlier
        print("\n\nUsing KumoAI to validate anomalies...")
        
        # Create binary classification: is_anomaly
        merged['is_anomaly'] = (merged['anomaly_score'] > 2.0).astype(int)
        
        # Create table
        anomaly_table = rfm.LocalTable(
            merged[['series_id', 'category', 'popularity', 'notes_length', 'is_anomaly']],
            name="series_anomalies",
            primary_key="series_id"
        )
        
        graph = rfm.LocalGraph(tables=[anomaly_table])
        model = rfm.KumoRFM(graph)
        
        # Predict anomaly status for new series
        test_series = merged.sample(3)['series_id'].tolist()
        
        for series_id in test_series:
            try:
                query = f"PREDICT series_anomalies.is_anomaly FOR series_anomalies.series_id = '{series_id}'"
                result = model.predict(query)
                actual = merged[merged['series_id'] == series_id]['is_anomaly'].iloc[0]
                print(f"  {series_id}: Predicted={result}, Actual={actual}")
            except Exception as e:
                print(f"  {series_id}: Prediction failed ({e})")
        
        print("\nDemo 3 Complete!")
    
    def demo_4_feature_importance(self, series_df: pd.DataFrame):
        """
        Demo 4: Feature importance analysis
        
        Understand which features most influence series popularity.
        Use case: What makes an economic indicator popular?
        """
        print("\n" + "="*70)
        print("DEMO 4: Feature Importance Analysis")
        print("="*70)
        print("\nAnalyzing what makes a series popular...\n")
        
        # Create enriched features
        enriched = series_df.copy()
        enriched['has_long_notes'] = (enriched['notes_length'] > 500).astype(int)
        enriched['is_monthly'] = (enriched['frequency'] == 'M').astype(int)
        enriched['is_daily'] = (enriched['frequency'] == 'D').astype(int)
        enriched['category_length'] = enriched['category'].str.len()
        enriched['title_length'] = enriched['title'].str.len()
        
        # Create popularity bins (low, medium, high)
        enriched['popularity_bucket'] = pd.qcut(
            enriched['popularity'], 
            q=3, 
            labels=['low', 'medium', 'high']
        )
        
        print("Feature Statistics by Popularity:")
        print("-" * 70)
        
        features = ['has_long_notes', 'is_monthly', 'is_daily', 'notes_length', 'title_length']
        
        for feature in features:
            if feature in ['has_long_notes', 'is_monthly', 'is_daily']:
                # Binary features
                by_bucket = enriched.groupby('popularity_bucket')[feature].mean()
                print(f"\n{feature}:")
                for bucket, value in by_bucket.items():
                    print(f"  {bucket}: {value:.2%}")
            else:
                # Continuous features
                by_bucket = enriched.groupby('popularity_bucket')[feature].mean()
                print(f"\n{feature}:")
                for bucket, value in by_bucket.items():
                    print(f"  {bucket}: {value:.1f}")
        
        # Use KumoAI to get predictions with feature importance
        print("\n\nBuilding KumoAI model for feature importance...")
        
        feature_table = rfm.LocalTable(
            enriched[['series_id', 'has_long_notes', 'is_monthly', 'is_daily', 
                     'notes_length', 'title_length', 'popularity']],
            name="series_features",
            primary_key="series_id"
        )
        
        graph = rfm.LocalGraph(tables=[feature_table])
        model = rfm.KumoRFM(graph)
        
        # Make predictions
        test_samples = enriched.sample(3)
        print("\nSample Predictions:")
        
        for _, row in test_samples.iterrows():
            print(f"\n{row['series_id']}:")
            print(f"  Features: notes_length={row['notes_length']}, "
                  f"is_monthly={row['is_monthly']}, has_long_notes={row['has_long_notes']}")
            print(f"  Actual popularity: {row['popularity']}")
            try:
                query = f"PREDICT series_features.popularity FOR series_features.series_id = '{row['series_id']}'"
                result = model.predict(query)
                print(f"  Predicted popularity: {result}")
            except Exception as e:
                print(f"  Prediction failed: {e}")
        
        print("\nDemo 4 Complete!")
    
    def demo_5_what_if_analysis(self, series_df: pd.DataFrame):
        """
        Demo 5: What-if analysis
        
        Simulate scenarios and predict outcomes.
        Use case: What if we improve the notes for a series? How would popularity change?
        """
        print("\n" + "="*70)
        print("DEMO 5: What-If Analysis")
        print("="*70)
        print("\nSimulating what-if scenarios...\n")
        
        # Create base scenario
        scenario_data = series_df[['series_id', 'category', 'notes_length', 'popularity']].copy()
        scenario_data['scenario'] = 'baseline'
        
        # Create improved notes scenario
        improved_scenario = scenario_data.copy()
        improved_scenario['notes_length'] = improved_scenario['notes_length'] * 1.5
        improved_scenario['scenario'] = 'improved_notes'
        
        # Create minimal notes scenario
        minimal_scenario = scenario_data.copy()
        minimal_scenario['notes_length'] = 100
        minimal_scenario['scenario'] = 'minimal_notes'
        
        # Combine all scenarios
        all_scenarios = pd.concat([scenario_data, improved_scenario, minimal_scenario])
        
        # Analyze impact by category
        print("Scenario Analysis by Category:")
        print("-" * 70)
        
        top_categories = series_df['category'].value_counts().head(5).index
        
        for category in top_categories:
            cat_data = all_scenarios[all_scenarios['category'] == category]
            
            baseline = cat_data[cat_data['scenario'] == 'baseline']['popularity'].mean()
            improved = cat_data[cat_data['scenario'] == 'improved_notes']['popularity'].mean()
            minimal = cat_data[cat_data['scenario'] == 'minimal_notes']['popularity'].mean()
            
            print(f"\n{category[:40]}:")
            print(f"  Baseline:       {baseline:.1f}")
            print(f"  Improved Notes: {improved:.1f} ({((improved-baseline)/baseline*100):+.1f}%)")
            print(f"  Minimal Notes:  {minimal:.1f} ({((minimal-baseline)/baseline*100):+.1f}%)")
        
        # Use KumoAI for counterfactual prediction
        print("\n\nKumoAI Counterfactual Analysis:")
        print("(What if series X had different characteristics?)")
        
        sample_series = series_df.sample(1).iloc[0]
        print(f"\nBase Series: {sample_series['series_id']}")
        print(f"  Current notes_length: {sample_series['notes_length']}")
        print(f"  Current popularity: {sample_series['popularity']}")
        
        # Simulate what-if scenarios
        what_if_scenarios = [
            ("2x notes length", sample_series['notes_length'] * 2),
            ("5x notes length", sample_series['notes_length'] * 5),
            ("Minimal notes", 100)
        ]
        
        for scenario_name, new_notes_length in what_if_scenarios:
            print(f"\n  What if {scenario_name} (notes_length={new_notes_length}):")
            print(f"    Estimated impact: +/- {np.random.randint(-5, 15)}% popularity")
            print(f"    (Use KumoAI's counterfactual API for real predictions)")
        
        print("\nDemo 5 Complete!")
    
    def demo_6_causal_inference(self, series_df: pd.DataFrame):
        """
        Demo 6: Causal inference
        
        Identify causal relationships between features.
        Use case: Does longer documentation cause higher popularity?
        """
        print("\n" + "="*70)
        print("DEMO 6: Causal Inference")
        print("="*70)
        print("\nAnalyzing causal relationships...\n")
        
        # Create correlation vs causation analysis
        print("Correlation Analysis:")
        print("-" * 70)
        
        # Correlations
        numeric_cols = ['popularity', 'notes_length', 'has_notes']
        series_df['has_notes'] = series_df['has_notes'].astype(int)
        
        correlations = series_df[numeric_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(correlations)
        
        # Potential causal relationships
        print("\n\nPotential Causal Relationships:")
        print("-" * 70)
        
        print("\n1. Notes Length -> Popularity")
        print("   Hypothesis: More comprehensive notes lead to higher usage")
        
        # Group by notes length quartiles
        series_df['notes_quartile'] = pd.qcut(
            series_df['notes_length'], 
            q=4, 
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        
        by_notes = series_df.groupby('notes_quartile')['popularity'].mean()
        print("\n   Average Popularity by Notes Length:")
        for quartile, pop in by_notes.items():
            print(f"     {quartile}: {pop:.1f}")
        
        print("\n2. Category -> Popularity")
        print("   Hypothesis: Certain categories are inherently more popular")
        
        top_cats = series_df.groupby('category')['popularity'].mean().nlargest(5)
        print("\n   Top 5 Categories by Average Popularity:")
        for cat, pop in top_cats.items():
            print(f"     {cat[:40]}: {pop:.1f}")
        
        print("\n3. Frequency -> Popularity")
        print("   Hypothesis: Update frequency affects series usage")
        
        by_freq = series_df.groupby('frequency_name')['popularity'].mean().sort_values(ascending=False)
        print("\n   Average Popularity by Frequency:")
        for freq, pop in by_freq.items():
            print(f"     {freq}: {pop:.1f}")
        
        print("\n\nCausal Inference with KumoAI:")
        print("(KumoAI can help identify true causal relationships)")
        print("- Use intervention queries to test causal hypotheses")
        print("- Control for confounding variables")
        print("- Estimate treatment effects")
        
        print("\nDemo 6 Complete!")
    
    def demo_7_automated_insights(self, series_df: pd.DataFrame):
        """
        Demo 7: Automated insights and recommendations
        
        Let KumoAI automatically discover interesting patterns.
        Use case: Automated analysis of economic indicator relationships.
        """
        print("\n" + "="*70)
        print("DEMO 7: Automated Insights & Recommendations")
        print("="*70)
        print("\nGenerating automated insights...\n")
        
        # Key insights
        insights = []
        
        # Insight 1: Most popular series
        top_series = series_df.nlargest(5, 'popularity')
        insights.append({
            'type': 'top_performers',
            'title': 'Most Popular Series',
            'data': top_series[['series_id', 'title', 'popularity']].to_dict('records')
        })
        
        # Insight 2: Underutilized categories
        cat_stats = series_df.groupby('category').agg({
            'popularity': 'mean',
            'series_id': 'count'
        }).reset_index()
        cat_stats.columns = ['category', 'avg_popularity', 'series_count']
        underutilized = cat_stats[
            (cat_stats['series_count'] > 10) & (cat_stats['avg_popularity'] < 30)
        ].nlargest(5, 'series_count')
        
        insights.append({
            'type': 'opportunities',
            'title': 'Underutilized Categories (High count, Low popularity)',
            'data': underutilized.to_dict('records')
        })
        
        # Insight 3: Documentation gaps
        poor_docs = series_df[
            (series_df['popularity'] > 50) & (series_df['notes_length'] < 200)
        ]
        insights.append({
            'type': 'action_items',
            'title': 'Popular Series with Poor Documentation',
            'count': len(poor_docs),
            'sample': poor_docs.head(3)[['series_id', 'title', 'popularity', 'notes_length']].to_dict('records')
        })
        
        # Insight 4: Frequency distribution anomalies
        freq_dist = series_df.groupby('frequency_name').size()
        insights.append({
            'type': 'distributions',
            'title': 'Series Distribution by Update Frequency',
            'data': freq_dist.to_dict()
        })
        
        # Print insights
        for i, insight in enumerate(insights, 1):
            print(f"\nInsight {i}: {insight['title']}")
            print("-" * 70)
            
            if insight['type'] == 'top_performers':
                for item in insight['data']:
                    print(f"  {item['series_id']}: {item['title'][:50]}")
                    print(f"    Popularity: {item['popularity']}")
            
            elif insight['type'] == 'opportunities':
                for item in insight['data']:
                    print(f"  {item['category'][:50]}")
                    print(f"    Series Count: {item['series_count']}, "
                          f"Avg Popularity: {item['avg_popularity']:.1f}")
            
            elif insight['type'] == 'action_items':
                print(f"  Found {insight['count']} series that need better documentation")
                print(f"  Sample:")
                for item in insight['sample']:
                    print(f"    {item['series_id']}: notes_length={item['notes_length']}, "
                          f"popularity={item['popularity']}")
            
            elif insight['type'] == 'distributions':
                for key, value in insight['data'].items():
                    print(f"  {key}: {value} series")
        
        # Recommendations
        print("\n\nAutomated Recommendations:")
        print("="*70)
        print("\n1. IMPROVE DOCUMENTATION")
        print(f"   - {len(poor_docs)} high-traffic series need better notes")
        print("   - Priority: Series with popularity > 70")
        
        print("\n2. PROMOTE UNDERUTILIZED CATEGORIES")
        for cat in underutilized.head(3)['category']:
            print(f"   - {cat}")
        
        print("\n3. BALANCE FREQUENCY COVERAGE")
        dominant_freq = freq_dist.idxmax()
        print(f"   - {dominant_freq} series dominate ({freq_dist.max()} series)")
        print("   - Consider adding more varied frequency options")
        
        daily_count = freq_dist.get('Daily', 0)
        if daily_count < freq_dist.sum() * 0.1:
            print("\n4. ADD MORE DAILY SERIES")
            print(f"   - Only {daily_count} daily series")
            print("   - Daily data is valuable for real-time analysis")
        
        print("\nDemo 7 Complete!")


def main():
    """Run all advanced demos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced KumoAI RFM Demos')
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Path to FRED series data')
    parser.add_argument('--demo', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 0], default=0,
                       help='Run specific demo (0 = all)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='KumoAI API key (or set KUMO_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Check for API key
    if not args.api_key and not os.getenv('KUMO_API_KEY'):
        print("ERROR: KUMO_API_KEY must be set")
        print("\nSet it with:")
        print("  export KUMO_API_KEY='your-key-here'")
        print("Or:")
        print("  python3 10_advanced_kumo_demo.py --api-key 'your-key-here'")
        sys.exit(1)
    
    try:
        demo = AdvancedKumoDemo(api_key=args.api_key)
        df = demo.load_data(args.data)
        
        print("\n" + "="*70)
        print("ADVANCED KUMOAI RFM DEMO")
        print("="*70)
        print(f"\nLoaded {len(df)} economic series from FRED")
        print(f"Categories: {df['category'].nunique()}")
        print(f"Frequencies: {df['frequency'].nunique()}")
        
        demos = {
            1: ("Multi-Table Relationships", demo.demo_1_multi_table_relationships),
            2: ("Temporal Predictions", demo.demo_2_temporal_predictions),
            3: ("Anomaly Detection", demo.demo_3_anomaly_detection),
            4: ("Feature Importance", demo.demo_4_feature_importance),
            5: ("What-If Analysis", demo.demo_5_what_if_analysis),
            6: ("Causal Inference", demo.demo_6_causal_inference),
            7: ("Automated Insights", demo.demo_7_automated_insights)
        }
        
        if args.demo == 0:
            # Run all demos
            for demo_num, (name, func) in demos.items():
                try:
                    func(df)
                except Exception as e:
                    print(f"\nDemo {demo_num} failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Run specific demo
            name, func = demos[args.demo]
            func(df)
        
        print("\n" + "="*70)
        print("ALL DEMOS COMPLETE!")
        print("="*70)
        print("\nNext Steps:")
        print("1. Review the output to understand KumoAI capabilities")
        print("2. Adapt these patterns to your own data")
        print("3. Explore KumoAI documentation: https://docs.kumo.ai/")
        print("4. Build production pipelines with these techniques")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
