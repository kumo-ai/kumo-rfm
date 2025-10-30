#!/usr/bin/env python3
"""
Example: Combining NetworkX and KumoAI for Economic Series Analysis

This script demonstrates how to:
1. Use KumoAI to discover relationships and make predictions
2. Use NetworkX to visualize those relationships
3. Combine both for powerful analytical workflows
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Import our visualization module
import sys
sys.path.insert(0, '/home/david/Desktop/kumo-rfm/projects/fred/map')
import importlib.util

# Load 08_network_visualizations module
spec_08 = importlib.util.spec_from_file_location("network_visualizations", "08_network_visualizations.py")
network_viz_module = importlib.util.module_from_spec(spec_08)
spec_08.loader.exec_module(network_viz_module)
FREDNetworkVisualizer = network_viz_module.FREDNetworkVisualizer

# Load 05_kumo_rfm_integration module
spec_05 = importlib.util.spec_from_file_location("kumo_integration", "05_kumo_rfm_integration.py")
kumo_module = importlib.util.module_from_spec(spec_05)
spec_05.loader.exec_module(kumo_module)
KumoFREDIntegration = kumo_module.KumoFREDIntegration


def example_1_basic_network_viz():
    """Example 1: Basic network visualization without KumoAI"""
    print("=" * 60)
    print("Example 1: Basic NetworkX Visualization")
    print("=" * 60)
    
    # Load data
    viz = FREDNetworkVisualizer(output_dir="visualizations")
    df = viz.load_data("data/fred_series_metadata.parquet")
    
    print(f"\nLoaded {len(df)} economic series")
    
    # Create category network
    print("\n Creating category network...")
    viz.visualize_category_network(df, layout='spring', save=True)
    
    # Create series similarity network
    print("\n Creating series similarity network...")
    viz.visualize_series_network(
        df, 
        embeddings=None,  # Will use category/frequency similarity
        similarity_threshold=0.6,
        max_nodes=30,
        layout='spring',
        save=True
    )
    
    print("\n Example 1 complete!")


def example_2_with_embeddings():
    """Example 2: Network visualization with vector embeddings"""
    print("\n" + "=" * 60)
    print("Example 2: Network with Semantic Embeddings")
    print("=" * 60)
    
    viz = FREDNetworkVisualizer(output_dir="visualizations")
    df = viz.load_data("data/fred_series_metadata.parquet")
    
    # Load embeddings if they exist
    embeddings_path = Path("data/embeddings/embeddings.npy")
    if embeddings_path.exists():
        print(f"\n Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        print(f"   Shape: {embeddings.shape}")
        
        # Create series network with embeddings
        print("\n Creating series network with semantic similarity...")
        viz.visualize_series_network(
            df,
            embeddings=embeddings,
            similarity_threshold=0.75,  # Higher threshold for embeddings
            max_nodes=50,
            layout='spring',
            save=True
        )
        
        # Community detection
        print("\n Detecting communities based on semantic similarity...")
        viz.visualize_community_detection(
            df,
            embeddings=embeddings,
            max_nodes=100,
            save=True
        )
        
        # Centrality analysis
        print("\n Analyzing node centrality...")
        viz.visualize_centrality_analysis(
            df,
            embeddings=embeddings,
            max_nodes=80,
            save=True
        )
        
    else:
        print(f"\n  Embeddings not found at {embeddings_path}")
        print("   Run: python3 04_vector_search.py --create")
    
    print("\n Example 2 complete!")


def example_3_kumo_predictions():
    """Example 3: Use KumoAI for predictions"""
    print("\n" + "=" * 60)
    print("Example 3: KumoAI Predictive Queries")
    print("=" * 60)
    
    # Check if KUMO_API_KEY is set
    import os
    if not os.getenv('KUMO_API_KEY'):
        print("\n  KUMO_API_KEY not set")
        print("   Export your API key: export KUMO_API_KEY='your-key'")
        print("   Skipping KumoAI examples...")
        return
    
    try:
        # Initialize KumoAI integration
        kumo = KumoFREDIntegration()
        df = kumo.load_data("data/fred_series_metadata.parquet")
        
        print(f"\n Building Kumo RFM graph...")
        kumo.build_graph(df)
        
        # Example prediction: Predict popularity for a series
        sample_series = df.sample(1).iloc[0]
        series_id = sample_series['series_id']
        
        print(f"\n Predicting popularity for series: {series_id}")
        print(f"   Title: {sample_series['title']}")
        print(f"   Actual popularity: {sample_series['popularity']}")
        
        result = kumo.predict_series_popularity(series_id)
        print(f"   Predicted result:")
        print(result)
        
        # Get recommendations
        print(f"\n Getting recommendations for 'inflation' analysis...")
        recommendations = kumo.recommend_series_for_analysis(df, 'inflation', top_k=5)
        print(recommendations)
        
    except Exception as e:
        print(f"\n KumoAI error: {e}")
        print("   Make sure kumoai is installed: pip install kumoai")
    
    print("\n Example 3 complete!")


def example_4_combined_workflow():
    """Example 4: Combined NetworkX + KumoAI workflow"""
    print("\n" + "=" * 60)
    print("Example 4: Combined NetworkX + KumoAI Workflow")
    print("=" * 60)
    
    # Load data
    viz = FREDNetworkVisualizer(output_dir="visualizations")
    df = viz.load_data("data/fred_series_metadata.parquet")
    
    # Step 1: Find central nodes in the network
    print("\n Step 1: Finding central economic indicators...")
    
    embeddings_path = Path("data/embeddings/embeddings.npy")
    embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
    
    # Build network
    G = viz.build_series_network(
        df,
        embeddings=embeddings,
        similarity_threshold=0.6,
        max_nodes=100
    )
    
    # Calculate centrality
    import networkx as nx
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Get top central nodes
    top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n Top 10 most connected series:")
    for i, (series_id, centrality) in enumerate(top_central, 1):
        series_info = df[df['series_id'] == series_id].iloc[0]
        print(f"   {i}. {series_id} (centrality: {centrality:.3f})")
        print(f"      {series_info['title'][:70]}")
        print(f"      Category: {series_info['category']}, Popularity: {series_info['popularity']}")
    
    # Step 2: Use KumoAI to predict relationships
    import os
    if os.getenv('KUMO_API_KEY'):
        try:
            print("\n Step 2: Using KumoAI to analyze top indicators...")
            
            kumo = KumoFREDIntegration()
            kumo.build_graph(df)
            
            # Predict popularity for top central nodes
            top_series_ids = [sid for sid, _ in top_central[:3]]
            
            print(f"\n Batch prediction for top 3 central series...")
            predictions = kumo.batch_predict_popularity(top_series_ids)
            print(predictions)
            
        except Exception as e:
            print(f"\n  KumoAI analysis skipped: {e}")
    else:
        print("\n  KUMO_API_KEY not set, skipping KumoAI analysis")
    
    # Step 3: Visualize the combined insights
    print("\n Step 3: Creating final visualization...")
    viz.visualize_centrality_analysis(
        df,
        embeddings=embeddings,
        max_nodes=100,
        save=True
    )
    
    print("\n Example 4 complete!")


def example_5_export_for_web():
    """Example 5: Export network to JSON for web visualization"""
    print("\n" + "=" * 60)
    print("Example 5: Export Network for Web Visualization")
    print("=" * 60)
    
    viz = FREDNetworkVisualizer(output_dir="visualizations")
    df = viz.load_data("data/fred_series_metadata.parquet")
    
    # Build category network
    print("\n Building category network...")
    G = viz.build_category_network(df)
    
    # Export to JSON (for D3.js, Cytoscape.js, etc.)
    output_path = "visualizations/category_network.json"
    viz.export_graph_to_json(output_path)
    
    print(f"\n Network exported to {output_path}")
    print("   You can use this with D3.js, Cytoscape.js, or other web frameworks")
    
    # Also create a simple HTML example
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>FRED Category Network</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #graph { width: 100vw; height: 100vh; }
        .node { stroke: #fff; stroke-width: 1.5px; }
        .link { stroke: #999; stroke-opacity: 0.6; }
    </style>
</head>
<body>
    <div id="graph"></div>
    <script>
        // Load the network data
        d3.json('category_network.json').then(data => {
            // D3.js force-directed graph code here
            console.log('Network loaded:', data);
            // Add your D3.js visualization code...
        });
    </script>
</body>
</html>"""
    
    html_path = "visualizations/network_viewer.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"   HTML viewer template: {html_path}")
    
    print("\n Example 5 complete!")


def main():
    """Run all examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NetworkX + KumoAI Examples')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5, 0], default=0,
                       help='Run specific example (0 = all)')
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_network_viz,
        2: example_2_with_embeddings,
        3: example_3_kumo_predictions,
        4: example_4_combined_workflow,
        5: example_5_export_for_web
    }
    
    if args.example == 0:
        # Run all examples
        for i in [1, 2, 3, 4, 5]:
            try:
                examples[i]()
            except Exception as e:
                print(f"\n Example {i} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Run specific example
        examples[args.example]()


if __name__ == '__main__':
    main()
