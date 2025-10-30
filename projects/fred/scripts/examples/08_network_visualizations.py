#!/usr/bin/env python3
"""
Network Visualizations for FRED Economic Series
Combines NetworkX and KumoAI to visualize relationships between economic indicators
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

try:
    import kumoai.experimental.rfm as rfm
    KUMO_AVAILABLE = True
except ImportError:
    KUMO_AVAILABLE = False
    print("Warning: kumoai not installed. Some features will be limited.")


class FREDNetworkVisualizer:
    """Network visualization for FRED economic series using NetworkX and KumoAI."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 9
        
        self.graph = nx.Graph()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load FRED series metadata."""
        if data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def build_category_network(self, series_df: pd.DataFrame) -> nx.Graph:
        """
        Build a network graph where:
        - Nodes = categories
        - Edge weight = number of shared characteristics
        """
        G = nx.Graph()
        
        # Add category nodes with attributes
        category_stats = series_df.groupby('category').agg({
            'series_id': 'count',
            'popularity': 'mean',
            'notes_length': 'mean'
        }).reset_index()
        
        for _, row in category_stats.iterrows():
            G.add_node(
                row['category'],
                size=row['series_id'],
                popularity=row['popularity'],
                notes_length=row['notes_length']
            )
        
        # Add edges based on shared frequencies
        categories = series_df['category'].unique()
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                # Find shared frequencies
                freq1 = set(series_df[series_df['category'] == cat1]['frequency'])
                freq2 = set(series_df[series_df['category'] == cat2]['frequency'])
                shared = len(freq1 & freq2)
                
                if shared > 0:
                    G.add_edge(cat1, cat2, weight=shared)
        
        self.graph = G
        return G
    
    def build_series_network(self, 
                           series_df: pd.DataFrame, 
                           embeddings: Optional[np.ndarray] = None,
                           similarity_threshold: float = 0.7,
                           max_nodes: int = 100) -> nx.Graph:
        """
        Build a network graph where:
        - Nodes = series
        - Edges = high similarity (based on embeddings or category/frequency)
        
        Args:
            series_df: DataFrame with series metadata
            embeddings: Optional embeddings array for similarity calculation
            similarity_threshold: Minimum similarity to create an edge
            max_nodes: Maximum number of nodes to include (uses top by popularity)
        """
        G = nx.Graph()
        
        # Select top series by popularity
        top_series = series_df.nlargest(max_nodes, 'popularity')
        
        # Add nodes
        for _, row in top_series.iterrows():
            G.add_node(
                row['series_id'],
                title=row['title'][:50],  # Truncate for display
                category=row['category'],
                frequency=row['frequency'],
                popularity=row['popularity']
            )
        
        # Add edges based on similarity
        if embeddings is not None:
            # Use embeddings for similarity
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get embeddings for top series
            top_indices = top_series.index.values
            top_embeddings = embeddings[top_indices]
            
            # Calculate similarity matrix
            sim_matrix = cosine_similarity(top_embeddings)
            
            # Add edges for high similarity
            for i, series1 in enumerate(top_series['series_id']):
                for j, series2 in enumerate(top_series['series_id']):
                    if i < j and sim_matrix[i, j] > similarity_threshold:
                        G.add_edge(series1, series2, weight=float(sim_matrix[i, j]))
        else:
            # Use category and frequency for similarity
            for i, row1 in top_series.iterrows():
                for j, row2 in top_series.iterrows():
                    if i < j:
                        # Same category = strong connection
                        # Same frequency = moderate connection
                        weight = 0
                        if row1['category'] == row2['category']:
                            weight += 0.6
                        if row1['frequency'] == row2['frequency']:
                            weight += 0.4
                        
                        if weight >= similarity_threshold:
                            G.add_edge(row1['series_id'], row2['series_id'], weight=weight)
        
        self.graph = G
        return G
    
    def visualize_category_network(self, 
                                  series_df: pd.DataFrame,
                                  layout: str = 'spring',
                                  save: bool = True) -> None:
        """
        Visualize category network with NetworkX.
        
        Args:
            series_df: DataFrame with series metadata
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            save: Whether to save the figure
        """
        G = self.build_category_network(series_df)
        
        print(f"Category network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Node sizes based on number of series
        node_sizes = [G.nodes[node]['size'] * 20 for node in G.nodes()]
        
        # Node colors based on average popularity
        node_colors = [G.nodes[node]['popularity'] for node in G.nodes()]
        
        # Edge widths based on weight
        edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='viridis',
            alpha=0.8,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.3,
            edge_color='gray',
            ax=ax
        )
        
        # Draw labels (truncate long category names)
        labels = {node: node[:30] for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title('FRED Category Network\nNode size = # of series, Color = avg popularity, Edge width = shared frequencies',
                    fontsize=14, pad=20)
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average Popularity', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"category_network_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved category network to {filepath}")
        
        plt.show()
        plt.close()
    
    def visualize_series_network(self,
                                series_df: pd.DataFrame,
                                embeddings: Optional[np.ndarray] = None,
                                similarity_threshold: float = 0.7,
                                max_nodes: int = 50,
                                layout: str = 'spring',
                                save: bool = True) -> None:
        """
        Visualize series similarity network.
        
        Args:
            series_df: DataFrame with series metadata
            embeddings: Optional embeddings for similarity
            similarity_threshold: Minimum similarity for edge
            max_nodes: Maximum nodes to display
            layout: Layout algorithm
            save: Whether to save the figure
        """
        G = self.build_series_network(series_df, embeddings, similarity_threshold, max_nodes)
        
        print(f"Series network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        if G.number_of_nodes() == 0:
            print("No nodes in graph. Try lowering similarity_threshold or increasing max_nodes.")
            return
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Node attributes
        node_sizes = [G.nodes[node]['popularity'] * 5 for node in G.nodes()]
        
        # Color by category
        categories = list(set(G.nodes[node]['category'] for node in G.nodes()))
        category_colors = {cat: i for i, cat in enumerate(categories)}
        node_colors = [category_colors[G.nodes[node]['category']] for node in G.nodes()]
        
        # Edge widths
        if G.number_of_edges() > 0:
            edge_widths = [G[u][v].get('weight', 1) * 3 for u, v in G.edges()]
        else:
            edge_widths = []
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='tab20',
            alpha=0.8,
            ax=ax
        )
        
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                alpha=0.2,
                edge_color='gray',
                ax=ax
            )
        
        # Draw labels (series IDs)
        nx.draw_networkx_labels(
            G, pos,
            labels={node: node for node in G.nodes()},
            font_size=7,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title(f'FRED Series Similarity Network (top {max_nodes} by popularity)\n' +
                    f'Node size = popularity, Color = category, Edge = similarity > {similarity_threshold}',
                    fontsize=14, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"series_network_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved series network to {filepath}")
        
        plt.show()
        plt.close()
    
    def visualize_community_detection(self,
                                     series_df: pd.DataFrame,
                                     embeddings: Optional[np.ndarray] = None,
                                     max_nodes: int = 100,
                                     save: bool = True) -> None:
        """
        Detect and visualize communities in the series network.
        
        Args:
            series_df: DataFrame with series metadata
            embeddings: Optional embeddings for similarity
            max_nodes: Maximum nodes to include
            save: Whether to save the figure
        """
        G = self.build_series_network(series_df, embeddings, similarity_threshold=0.5, max_nodes=max_nodes)
        
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print("Graph too sparse for community detection. Try different parameters.")
            return
        
        print(f"Detecting communities in network with {G.number_of_nodes()} nodes...")
        
        # Detect communities using Louvain algorithm
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G)
        
        print(f"Found {len(communities)} communities")
        
        # Assign community IDs to nodes
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i
        
        # Layout
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Node attributes
        node_sizes = [G.nodes[node]['popularity'] * 5 for node in G.nodes()]
        node_colors = [node_to_community[node] for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='tab20',
            alpha=0.8,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.2,
            edge_color='gray',
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            labels={node: node for node in G.nodes()},
            font_size=6,
            ax=ax
        )
        
        ax.set_title(f'FRED Series Communities (detected {len(communities)} clusters)\n' +
                    f'Node size = popularity, Color = community',
                    fontsize=14, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"community_detection_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved community detection to {filepath}")
            
            # Save community assignments
            community_data = []
            for i, comm in enumerate(communities):
                for node in comm:
                    community_data.append({
                        'series_id': node,
                        'community': i,
                        'community_size': len(comm)
                    })
            
            comm_df = pd.DataFrame(community_data)
            comm_filepath = self.output_dir / f"communities_{timestamp}.csv"
            comm_df.to_csv(comm_filepath, index=False)
            print(f" Saved community assignments to {comm_filepath}")
        
        plt.show()
        plt.close()
    
    def visualize_centrality_analysis(self,
                                     series_df: pd.DataFrame,
                                     embeddings: Optional[np.ndarray] = None,
                                     max_nodes: int = 100,
                                     save: bool = True) -> None:
        """
        Analyze and visualize node centrality measures.
        
        Args:
            series_df: DataFrame with series metadata
            embeddings: Optional embeddings for similarity
            max_nodes: Maximum nodes to include
            save: Whether to save the figure
        """
        G = self.build_series_network(series_df, embeddings, similarity_threshold=0.5, max_nodes=max_nodes)
        
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print("Graph too sparse for centrality analysis.")
            return
        
        print(f"Analyzing centrality for {G.number_of_nodes()} nodes...")
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # 1. Degree centrality
        node_colors = [degree_centrality[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='YlOrRd', 
                              node_size=300, alpha=0.8, ax=axes[0, 0])
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=axes[0, 0])
        nx.draw_networkx_labels(G, pos, font_size=6, ax=axes[0, 0])
        axes[0, 0].set_title('Degree Centrality\n(How many connections)', fontsize=12)
        axes[0, 0].axis('off')
        
        # 2. Betweenness centrality
        node_colors = [betweenness_centrality[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='YlGnBu',
                              node_size=300, alpha=0.8, ax=axes[0, 1])
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=axes[0, 1])
        nx.draw_networkx_labels(G, pos, font_size=6, ax=axes[0, 1])
        axes[0, 1].set_title('Betweenness Centrality\n(Bridge between clusters)', fontsize=12)
        axes[0, 1].axis('off')
        
        # 3. Closeness centrality
        node_colors = [closeness_centrality[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='PuRd',
                              node_size=300, alpha=0.8, ax=axes[1, 0])
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=axes[1, 0])
        nx.draw_networkx_labels(G, pos, font_size=6, ax=axes[1, 0])
        axes[1, 0].set_title('Closeness Centrality\n(Close to all nodes)', fontsize=12)
        axes[1, 0].axis('off')
        
        # 4. Top 10 central nodes
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        axes[1, 1].axis('off')
        text = "Top 10 Most Central Series\n\n"
        text += "By Degree (Connections):\n"
        for i, (node, score) in enumerate(top_degree, 1):
            text += f"{i}. {node}: {score:.3f}\n"
        
        text += "\nBy Betweenness (Bridges):\n"
        for i, (node, score) in enumerate(top_betweenness, 1):
            text += f"{i}. {node}: {score:.3f}\n"
        
        axes[1, 1].text(0.1, 0.9, text, transform=axes[1, 1].transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Network Centrality Analysis', fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"centrality_analysis_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved centrality analysis to {filepath}")
            
            # Save centrality data
            centrality_data = pd.DataFrame({
                'series_id': list(degree_centrality.keys()),
                'degree_centrality': list(degree_centrality.values()),
                'betweenness_centrality': list(betweenness_centrality.values()),
                'closeness_centrality': list(closeness_centrality.values())
            })
            data_filepath = self.output_dir / f"centrality_data_{timestamp}.csv"
            centrality_data.to_csv(data_filepath, index=False)
            print(f" Saved centrality data to {data_filepath}")
        
        plt.show()
        plt.close()
    
    def export_graph_to_json(self, filepath: str) -> None:
        """Export NetworkX graph to JSON format for web visualization (D3.js, etc)."""
        from networkx.readwrite import json_graph
        
        data = json_graph.node_link_data(self.graph)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f" Exported graph to {filepath}")


def main():
    """Demo of network visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FRED Network Visualizations')
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Path to FRED series data')
    parser.add_argument('--embeddings', type=str, default=None,
                       help='Path to embeddings .npy file (optional)')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory')
    parser.add_argument('--max-nodes', type=int, default=50,
                       help='Maximum nodes in series networks')
    parser.add_argument('--similarity', type=float, default=0.7,
                       help='Similarity threshold for edges')
    parser.add_argument('--viz', type=str, nargs='+', 
                       default=['category', 'series', 'community', 'centrality'],
                       choices=['category', 'series', 'community', 'centrality', 'all'],
                       help='Visualizations to create')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = FREDNetworkVisualizer(output_dir=args.output)
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = viz.load_data(args.data)
    print(f"Loaded {len(df)} series")
    
    # Load embeddings if provided
    embeddings = None
    if args.embeddings:
        print(f"Loading embeddings from {args.embeddings}...")
        embeddings = np.load(args.embeddings)
        print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Create visualizations
    viz_types = args.viz if 'all' not in args.viz else ['category', 'series', 'community', 'centrality']
    
    if 'category' in viz_types:
        print("\n=== Creating Category Network ===")
        viz.visualize_category_network(df, layout='spring', save=True)
    
    if 'series' in viz_types:
        print("\n=== Creating Series Network ===")
        viz.visualize_series_network(df, embeddings, args.similarity, args.max_nodes, 
                                     layout='spring', save=True)
    
    if 'community' in viz_types:
        print("\n=== Detecting Communities ===")
        viz.visualize_community_detection(df, embeddings, args.max_nodes, save=True)
    
    if 'centrality' in viz_types:
        print("\n=== Analyzing Centrality ===")
        viz.visualize_centrality_analysis(df, embeddings, args.max_nodes, save=True)
    
    print("\n All visualizations complete!")


if __name__ == '__main__':
    main()
