#!/usr/bin/env python3
"""
Kumo RFM CLI - Interactive PostgreSQL Database Management Tool

Modes:
- automate: Automatically populate database based on patterns and rules
- augment: AI-assisted SQL with autocomplete and recommendations
- alone: Direct SQL/psql command execution
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import readline
from typing import Optional, List, Tuple
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import networkx as nx
import json
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform color support
init(autoreset=True)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars


class KumoRFMCLI:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.mode = None
        self.history = []
        self.last_query_results = None  # Store last query results for visualization
        self.last_query = None  # Store last executed query
        self.graph = None  # NetworkX graph for relationship visualization
        
        # Setup visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def connect_postgres(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            # Try environment variables first
            db_params = {
                'host': os.getenv('PGHOST', 'localhost'),
                'port': os.getenv('PGPORT', '5432'),
                'database': os.getenv('PGDATABASE', 'postgres'),
                'user': os.getenv('PGUSER', 'postgres'),
                'password': os.getenv('PGPASSWORD', '')
            }
            
            # Prompt for missing credentials
            if not db_params['user']:
                db_params['user'] = input("PostgreSQL username: ")
            if not db_params['password']:
                from getpass import getpass
                db_params['password'] = getpass("PostgreSQL password: ")
            
            self.conn = psycopg2.connect(**db_params)
            self.cursor = self.conn.cursor()
            print(f"{Fore.GREEN}✓ Connected to PostgreSQL at {db_params['host']}:{db_params['port']}/{db_params['database']}{Style.RESET_ALL}")
            return True
            
        except psycopg2.Error as e:
            print(f"{Fore.RED}✗ Connection failed: {e}{Style.RESET_ALL}")
            return False
    
    def select_mode(self) -> str:
        """Display mode selection menu"""
        print("\n" + "="*60)
        print("Kumo RFM CLI - PostgreSQL Database Management")
        print("="*60)
        print("\nSelect a mode:")
        print("  1. augment   - AI-assisted SQL with recommendations")
        print("  2. alone     - Direct SQL/psql command execution")
        print("  3. visualize - Visualize query results")
        print("  4. exit      - Quit")
        print()
        
        while True:
            choice = input("Enter mode (1-4 or name): ").strip().lower()
            
            mode_map = {
                '1': 'augment', 'augment': 'augment',
                '2': 'alone', 'alone': 'alone',
                '3': 'visualize', 'visualize': 'visualize', 'viz': 'visualize',
                '4': 'exit', 'exit': 'exit', 'quit': 'exit'
            }
            
            if choice in mode_map:
                return mode_map[choice]
            else:
                print("Invalid choice. Please try again.")
    
    def run_augment_mode(self):
        """Augment mode - AI-assisted SQL with recommendations"""
        print("\n" + "-"*60)
        print("AUGMENT MODE")
        print("-"*60)
        print("AI-assisted SQL with autocomplete and recommendations.")
        print("Commands: help, suggest, viz, switch <mode>, back")
        print()
        
        while True:
            try:
                cmd = input("augment> ").strip()
                
                if not cmd:
                    continue
                elif cmd == 'back':
                    break
                elif cmd in ['viz', 'visualize']:
                    # Switch to visualize mode with current data
                    return 'visualize'
                elif cmd.startswith('switch '):
                    mode = cmd[7:].strip()
                    return mode if mode in ['alone', 'visualize', 'augment'] else None
                elif cmd == 'help':
                    self.show_augment_help()
                elif cmd == 'suggest':
                    self.suggest_queries()
                elif cmd.startswith('explain '):
                    query = cmd[8:].strip()
                    self.explain_query(query)
                elif cmd.startswith('optimize '):
                    query = cmd[9:].strip()
                    self.optimize_query(query)
                else:
                    # Treat as SQL query with augmentation
                    self.execute_with_augmentation(cmd)
                    
            except KeyboardInterrupt:
                print("\nUse 'back' to return to mode selection")
            except Exception as e:
                print(f"Error: {e}")
    
    def show_augment_help(self):
        """Show augment mode help"""
        print("\nAugment Mode Commands:")
        print("  help                - Show this help")
        print("  suggest             - Get query suggestions")
        print("  explain <query>     - Explain query execution plan")
        print("  optimize <query>    - Get optimization recommendations")
        print("  viz                 - Switch to visualize mode (with current data)")
        print("  switch <mode>       - Switch to another mode (alone/visualize)")
        print("  <any SQL>           - Execute with AI recommendations")
        print("  back                - Return to mode selection")
        print()
    
    def suggest_queries(self):
        """Suggest common queries based on schema"""
        try:
            # Get table list
            self.cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in self.cursor.fetchall()]
            
            if not tables:
                print("No tables found in database.")
                return
            
            print("\nSuggested queries based on your schema:")
            for table in tables[:5]:  # Show first 5 tables
                print(f"\n  Table: {table}")
                print(f"    SELECT * FROM {table} LIMIT 10;")
                print(f"    SELECT COUNT(*) FROM {table};")
                print(f"    SELECT * FROM {table} WHERE <condition>;")
            print()
            
        except psycopg2.Error as e:
            print(f"Error fetching schema: {e}")
    
    def explain_query(self, query: str):
        """Explain query execution plan"""
        try:
            explain_query = f"EXPLAIN ANALYZE {query}"
            self.cursor.execute(explain_query)
            results = self.cursor.fetchall()
            
            print("\nQuery Execution Plan:")
            for row in results:
                print(f"  {row[0]}")
            print()
            
        except psycopg2.Error as e:
            print(f"Error explaining query: {e}")
            self.conn.rollback()
    
    def optimize_query(self, query: str):
        """Provide optimization recommendations"""
        print("\nOptimization recommendations:")
        
        recommendations = []
        query_lower = query.lower()
        
        if 'select *' in query_lower:
            recommendations.append("• Replace SELECT * with specific column names")
        
        if 'where' not in query_lower and 'select' in query_lower:
            recommendations.append("• Consider adding WHERE clause to filter data")
        
        if 'limit' not in query_lower and 'select' in query_lower:
            recommendations.append("• Add LIMIT clause for large result sets")
        
        if 'join' in query_lower and 'where' in query_lower:
            recommendations.append("• Ensure JOIN conditions use indexed columns")
        
        if not recommendations:
            recommendations.append("• Query looks good! Run EXPLAIN ANALYZE to check performance")
        
        for rec in recommendations:
            print(f"  {rec}")
        print()
    
    def execute_with_augmentation(self, query: str):
        """Execute query with optional AI recommendations"""
        # Store query
        self.last_query = query
        
        # Just execute - no confirmation needed
        self.execute_sql(query)
    
    def run_alone_mode(self):
        """Alone mode - direct SQL execution"""
        print("\n" + "-"*60)
        print("ALONE MODE")
        print("-"*60)
        print("Direct SQL/psql command execution.")
        print("Commands: \\dt (tables), \\d <table> (describe), viz, switch <mode>, \\q (quit)")
        print()
        
        while True:
            try:
                query = input("sql> ").strip()
                
                if not query:
                    continue
                elif query in ['\\q', 'quit', 'exit', 'back']:
                    break
                elif query in ['viz', 'visualize']:
                    return 'visualize'
                elif query.startswith('switch '):
                    mode = query[7:].strip()
                    return mode if mode in ['augment', 'visualize', 'alone'] else None
                elif query == '\\dt':
                    self.list_tables()
                elif query.startswith('\\d '):
                    table = query[3:].strip()
                    self.describe_table(table)
                elif query == '\\l':
                    self.list_databases()
                else:
                    self.last_query = query
                    self.execute_sql(query)
                    
            except KeyboardInterrupt:
                print("\nUse '\\q' or 'back' to return to mode selection")
            except Exception as e:
                print(f"Error: {e}")
    
    def list_tables(self):
        """List all tables in current database"""
        try:
            self.cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = self.cursor.fetchall()
            
            print(f"\n{Fore.CYAN}Tables:{Style.RESET_ALL}")
            for table in tables:
                print(f"  {Fore.YELLOW}{table[0]}{Style.RESET_ALL}")
            print()
            
        except psycopg2.Error as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def describe_table(self, table_name: str):
        """Describe table structure"""
        try:
            self.cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            columns = self.cursor.fetchall()
            
            if not columns:
                print(f"{Fore.RED}Table '{table_name}' not found.{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.CYAN}Table: {Fore.YELLOW}{table_name}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'Column':<30} {'Type':<20} {'Nullable':<10}{Style.RESET_ALL}")
            print("-" * 60)
            for col in columns:
                print(f"{Fore.GREEN}{col[0]:<30}{Style.RESET_ALL} {col[1]:<20} {col[2]:<10}")
            print()
            
        except psycopg2.Error as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def list_databases(self):
        """List all databases"""
        try:
            self.cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false")
            databases = self.cursor.fetchall()
            
            print("\nDatabases:")
            for db in databases:
                print(f"  {db[0]}")
            print()
            
        except psycopg2.Error as e:
            print(f"Error: {e}")
    
    def execute_sql(self, query: str, store_results: bool = True):
        """Execute SQL query and display results"""
        try:
            self.cursor.execute(query)
            
            # Check if query returns results
            if self.cursor.description:
                columns = [desc[0] for desc in self.cursor.description]
                results = self.cursor.fetchall()
                
                # Store results for visualization
                if store_results and results:
                    self.last_query_results = pd.DataFrame(results, columns=columns)
                
                # Display results
                if results:
                    # Print header with color
                    header = " | ".join(f"{col:<20}" for col in columns)
                    print(f"\n{Fore.CYAN}{header}{Style.RESET_ALL}")
                    print("-" * len(header))
                    
                    # Print rows
                    for row in results[:100]:  # Limit to 100 rows
                        print(" | ".join(f"{str(val):<20}" for val in row))
                    
                    if len(results) > 100:
                        print(f"\n{Fore.YELLOW}... ({len(results) - 100} more rows){Style.RESET_ALL}")
                    
                    print(f"\n{Fore.GREEN}({len(results)} rows){Style.RESET_ALL}")
                    
                    if store_results:
                        print(f"\n{Fore.MAGENTA}Tip: Type 'viz' to visualize this data{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Query returned no rows.{Style.RESET_ALL}")
                    if store_results:
                        self.last_query_results = None
            else:
                self.conn.commit()
                print(f"{Fore.GREEN}✓ Query executed successfully. Rows affected: {self.cursor.rowcount}{Style.RESET_ALL}")
                if store_results:
                    self.last_query_results = None
            
            print()
            
        except psycopg2.Error as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            self.conn.rollback()
            if store_results:
                self.last_query_results = None
    
    def run_visualize_mode(self):
        """Visualize mode - create charts from query results"""
        print("\n" + "-"*60)
        print("VISUALIZE MODE")
        print("-"*60)
        print("Create visualizations from query results.")
        
        if self.last_query_results is not None:
            print(f"Data loaded: {len(self.last_query_results)} rows")
        else:
            print("No data loaded. Run a query first.")
        
        print("\nCommands: query <sql>, plot <type>, list, export <file>, switch <mode>, back")
        print("Plot types: bar, line, scatter, hist, pie, heatmap, box, graph")
        print()
        
        while True:
            try:
                cmd = input("viz> ").strip()
                
                if not cmd:
                    continue
                elif cmd == 'back':
                    break
                elif cmd.startswith('switch '):
                    mode = cmd[7:].strip()
                    return mode if mode in ['augment', 'alone', 'visualize'] else None
                elif cmd == 'help':
                    self.show_viz_help()
                elif cmd == 'list':
                    self.show_available_plots()
                elif cmd.startswith('query '):
                    query = cmd[6:].strip()
                    self.last_query = query
                    self.execute_sql(query, store_results=True)
                elif cmd.startswith('plot '):
                    plot_type = cmd[5:].strip()
                    self.create_plot(plot_type)
                elif cmd.startswith('export '):
                    filename = cmd[7:].strip()
                    self.export_results(filename)
                elif cmd == 'show':
                    self.show_current_data()
                elif cmd == 'rerun' and self.last_query:
                    print(f"Re-running: {self.last_query}")
                    self.execute_sql(self.last_query, store_results=True)
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'back' to return to mode selection")
            except Exception as e:
                print(f"Error: {e}")
    
    def show_viz_help(self):
        """Show visualize mode help"""
        print("\nVisualize Mode Commands:")
        print("  help                - Show this help")
        print("  query <sql>         - Run query and store results")
        print("  rerun               - Re-run last query")
        print("  show                - Show current data info")
        print("  list                - List available plot types")
        print("  plot <type>         - Create a plot (bar/line/scatter/hist/pie/etc)")
        print("  export <file>       - Export data to file (.csv/.json/.parquet)")
        print("  switch <mode>       - Switch to another mode (augment/alone)")
        print("  back                - Return to mode selection")
        print()
    
    def show_available_plots(self):
        """Show available plot types"""
        print("\nAvailable plot types:")
        print("  bar      - Bar chart (requires categorical x, numeric y)")
        print("  line     - Line chart (requires x and y columns)")
        print("  scatter  - Scatter plot (requires x and y columns)")
        print("  hist     - Histogram (requires numeric column)")
        print("  pie      - Pie chart (requires categorical and numeric columns)")
        print("  heatmap  - Heatmap (requires pivot-able data)")
        print("  box      - Box plot (requires numeric data)")
        print("  graph    - Network graph (requires node/edge data or relationships)")
        print()
    
    def show_current_data(self):
        """Show current query results"""
        if self.last_query_results is None:
            print("No query results available. Run a query first.")
            return
        
        print("\nCurrent data:")
        print(self.last_query_results.head(20))
        print(f"\nShape: {self.last_query_results.shape[0]} rows × {self.last_query_results.shape[1]} columns")
        print(f"Columns: {list(self.last_query_results.columns)}")
        print()
    
    def create_plot(self, plot_type: str):
        """Create a plot from stored query results"""
        if self.last_query_results is None or self.last_query_results.empty:
            print("No query results to visualize. Run a query first with: query <sql>")
            return
        
        df = self.last_query_results
        
        try:
            if plot_type == 'bar':
                self._plot_bar(df)
            elif plot_type == 'line':
                self._plot_line(df)
            elif plot_type == 'scatter':
                self._plot_scatter(df)
            elif plot_type == 'hist':
                self._plot_histogram(df)
            elif plot_type == 'pie':
                self._plot_pie(df)
            elif plot_type == 'heatmap':
                self._plot_heatmap(df)
            elif plot_type == 'box':
                self._plot_box(df)
            elif plot_type == 'graph':
                self._plot_graph(df)
            else:
                print(f"Unknown plot type: {plot_type}")
                print("Use 'list' to see available plot types")
        
        except Exception as e:
            print(f"Error creating plot: {e}")
            print("Make sure your data has the right format for this plot type.")
    
    def _plot_bar(self, df: pd.DataFrame):
        """Create a bar chart"""
        if len(df.columns) < 2:
            print("Bar chart requires at least 2 columns (x and y)")
            return
        
        x_col = df.columns[0]
        y_col = df.columns[1]
        
        print(f"Creating bar chart: {x_col} vs {y_col}")
        
        plt.figure(figsize=(12, 6))
        plt.bar(df[x_col].head(20), df[y_col].head(20))
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} by {x_col}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        print(" Chart displayed")
    
    def _plot_line(self, df: pd.DataFrame):
        """Create a line chart"""
        if len(df.columns) < 2:
            print("Line chart requires at least 2 columns (x and y)")
            return
        
        x_col = df.columns[0]
        y_cols = df.columns[1:]
        
        print(f"Creating line chart: {x_col} vs {', '.join(y_cols)}")
        
        plt.figure(figsize=(12, 6))
        for y_col in y_cols:
            plt.plot(df[x_col], df[y_col], marker='o', label=y_col)
        
        plt.xlabel(x_col)
        plt.ylabel('Value')
        plt.title(f"Line Chart: {', '.join(y_cols)} over {x_col}")
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(" Chart displayed")
    
    def _plot_scatter(self, df: pd.DataFrame):
        """Create a scatter plot"""
        if len(df.columns) < 2:
            print("Scatter plot requires at least 2 columns (x and y)")
            return
        
        x_col = df.columns[0]
        y_col = df.columns[1]
        
        print(f"Creating scatter plot: {x_col} vs {y_col}")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter: {x_col} vs {y_col}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(" Chart displayed")
    
    def _plot_histogram(self, df: pd.DataFrame):
        """Create a histogram"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns found for histogram")
            return
        
        col = numeric_cols[0]
        print(f"Creating histogram for: {col}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f"Distribution of {col}")
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        print(" Chart displayed")
    
    def _plot_pie(self, df: pd.DataFrame):
        """Create a pie chart"""
        if len(df.columns) < 2:
            print("Pie chart requires at least 2 columns (labels and values)")
            return
        
        label_col = df.columns[0]
        value_col = df.columns[1]
        
        print(f"Creating pie chart: {label_col} with values from {value_col}")
        
        # Take top 10 to avoid overcrowding
        data = df.nlargest(10, value_col)
        
        plt.figure(figsize=(10, 8))
        plt.pie(data[value_col], labels=data[label_col], autopct='%1.1f%%', startangle=90)
        plt.title(f"Distribution: {value_col} by {label_col}")
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        print(" Chart displayed")
    
    def _plot_heatmap(self, df: pd.DataFrame):
        """Create a heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric data found for heatmap")
            return
        
        print("Creating correlation heatmap")
        
        plt.figure(figsize=(10, 8))
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        
        print(" Chart displayed")
    
    def _plot_box(self, df: pd.DataFrame):
        """Create a box plot"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns found for box plot")
            return
        
        print(f"Creating box plot for: {', '.join(numeric_cols)}")
        
        plt.figure(figsize=(12, 6))
        df[numeric_cols].boxplot()
        plt.ylabel('Value')
        plt.title('Box Plot of Numeric Columns')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        print(" Chart displayed")
    
    def _plot_graph(self, df: pd.DataFrame):
        """Create a network graph visualization"""
        # Determine if data represents edges or needs to be converted
        if len(df.columns) >= 2:
            # Assume first two columns are source and target
            source_col = df.columns[0]
            target_col = df.columns[1]
            
            # Check for weight column
            weight_col = None
            if len(df.columns) >= 3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    weight_col = numeric_cols[0]
            
            print(f"Creating network graph: {source_col} -> {target_col}")
            if weight_col:
                print(f"  Using {weight_col} as edge weight")
            
            # Create graph
            G = nx.DiGraph() if self._is_directed(df, source_col, target_col) else nx.Graph()
            
            # Add edges
            for _, row in df.iterrows():
                if weight_col:
                    G.add_edge(row[source_col], row[target_col], weight=row[weight_col])
                else:
                    G.add_edge(row[source_col], row[target_col])
            
            self.graph = G
            
            # Visualize
            plt.figure(figsize=(14, 10))
            
            # Choose layout based on graph size
            if len(G.nodes()) < 50:
                pos = nx.spring_layout(G, k=1, iterations=50)
            else:
                pos = nx.kamada_kawai_layout(G)
            
            # Draw nodes
            node_sizes = [300 + 100 * G.degree(node) for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                  node_color='lightblue', alpha=0.7)
            
            # Draw edges
            if weight_col:
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                nx.draw_networkx_edges(G, pos, width=[w/max(weights)*3 for w in weights],
                                      alpha=0.5, edge_color='gray', arrows=True)
            else:
                nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            plt.title(f"Network Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Print graph statistics
            print(f"\n Graph Statistics:")
            print(f"  Nodes: {len(G.nodes())}")
            print(f"  Edges: {len(G.edges())}")
            print(f"  Density: {nx.density(G):.4f}")
            
            if nx.is_connected(G.to_undirected()):
                print(f"  Average shortest path: {nx.average_shortest_path_length(G.to_undirected()):.2f}")
            
            # Top nodes by degree
            top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n  Top 5 nodes by degree:")
            for node, degree in top_nodes:
                print(f"    {node}: {degree}")
        
        else:
            print("Graph visualization requires at least 2 columns (source, target)")
            print("Format: node1, node2, [optional: weight]")
    
    def _is_directed(self, df: pd.DataFrame, source_col: str, target_col: str) -> bool:
        """Determine if graph should be directed based on edge reciprocity"""
        edges = set(zip(df[source_col], df[target_col]))
        reverse_edges = set(zip(df[target_col], df[source_col]))
        
        # If less than 30% of edges are bidirectional, assume directed
        bidirectional = edges.intersection(reverse_edges)
        return len(bidirectional) / len(edges) < 0.3 if edges else False
    
    def export_results(self, filename: str):
        """Export query results to file"""
        if self.last_query_results is None or self.last_query_results.empty:
            print("No query results to export. Run a query first.")
            return
        
        try:
            if filename.endswith('.csv'):
                self.last_query_results.to_csv(filename, index=False)
            elif filename.endswith('.json'):
                self.last_query_results.to_json(filename, orient='records', indent=2)
            elif filename.endswith('.xlsx'):
                self.last_query_results.to_excel(filename, index=False)
            elif filename.endswith('.parquet'):
                self.last_query_results.to_parquet(filename, index=False)
            elif filename.endswith('.gexf') and self.graph is not None:
                # Export NetworkX graph
                nx.write_gexf(self.graph, filename)
            elif filename.endswith('.graphml') and self.graph is not None:
                nx.write_graphml(self.graph, filename)
            else:
                # Default to CSV
                if not '.' in filename:
                    filename += '.csv'
                self.last_query_results.to_csv(filename, index=False)
            
            print(f" Results exported to {filename}")
            
        except Exception as e:
            print(f"Error exporting results: {e}")
    
    def initialize_monolith_tables(self):
        """Create tables for tracking Kumo RFM use cases and metadata"""
        try:
            # Create schema for Kumo RFM tracking
            schema_sql = """
            -- Table for tracking RFM use cases
            CREATE TABLE IF NOT EXISTS kumo_rfm_use_cases (
                use_case_id SERIAL PRIMARY KEY,
                use_case_name VARCHAR(255) NOT NULL,
                description TEXT,
                category VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB,
                is_active BOOLEAN DEFAULT TRUE
            );
            
            -- Table for tracking RFM queries
            CREATE TABLE IF NOT EXISTS kumo_rfm_queries (
                query_id SERIAL PRIMARY KEY,
                use_case_id INTEGER REFERENCES kumo_rfm_use_cases(use_case_id),
                query_text TEXT NOT NULL,
                query_type VARCHAR(50), -- 'predict', 'discover', 'analyze'
                execution_time_ms INTEGER,
                result_count INTEGER,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            );
            
            -- Table for tracking predictions
            CREATE TABLE IF NOT EXISTS kumo_rfm_predictions (
                prediction_id SERIAL PRIMARY KEY,
                query_id INTEGER REFERENCES kumo_rfm_queries(query_id),
                target_table VARCHAR(255),
                target_column VARCHAR(255),
                prediction_value TEXT,
                confidence_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            );
            
            -- Table for tracking graph relationships
            CREATE TABLE IF NOT EXISTS kumo_rfm_relationships (
                relationship_id SERIAL PRIMARY KEY,
                use_case_id INTEGER REFERENCES kumo_rfm_use_cases(use_case_id),
                source_entity VARCHAR(255) NOT NULL,
                target_entity VARCHAR(255) NOT NULL,
                relationship_type VARCHAR(100),
                strength FLOAT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            );
            
            -- Table for tracking model performance
            CREATE TABLE IF NOT EXISTS kumo_rfm_performance (
                performance_id SERIAL PRIMARY KEY,
                use_case_id INTEGER REFERENCES kumo_rfm_use_cases(use_case_id),
                metric_name VARCHAR(100) NOT NULL,
                metric_value FLOAT NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            );
            
            -- Indexes for better query performance
            CREATE INDEX IF NOT EXISTS idx_use_cases_category ON kumo_rfm_use_cases(category);
            CREATE INDEX IF NOT EXISTS idx_queries_use_case ON kumo_rfm_queries(use_case_id);
            CREATE INDEX IF NOT EXISTS idx_queries_executed ON kumo_rfm_queries(executed_at);
            CREATE INDEX IF NOT EXISTS idx_predictions_query ON kumo_rfm_predictions(query_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_use_case ON kumo_rfm_relationships(use_case_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON kumo_rfm_relationships(source_entity);
            CREATE INDEX IF NOT EXISTS idx_relationships_target ON kumo_rfm_relationships(target_entity);
            """
            
            # Execute schema creation
            for statement in schema_sql.split(';'):
                if statement.strip():
                    self.cursor.execute(statement)
            
            self.conn.commit()
            print(" Kumo RFM monolith tables initialized successfully")
            return True
            
        except psycopg2.Error as e:
            print(f"Error initializing monolith tables: {e}")
            self.conn.rollback()
            return False
    
    def save_use_case(self, name: str, description: str, category: str, metadata: dict = None):
        """Save a new Kumo RFM use case to the database"""
        try:
            query = """
                INSERT INTO kumo_rfm_use_cases (use_case_name, description, category, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING use_case_id;
            """
            
            self.cursor.execute(query, (
                name,
                description,
                category,
                json.dumps(metadata) if metadata else None
            ))
            
            use_case_id = self.cursor.fetchone()[0]
            self.conn.commit()
            
            print(f" Use case saved with ID: {use_case_id}")
            return use_case_id
            
        except psycopg2.Error as e:
            print(f"Error saving use case: {e}")
            self.conn.rollback()
            return None
    
    def log_query_execution(self, use_case_id: int, query_text: str, query_type: str, 
                           execution_time_ms: int, result_count: int, metadata: dict = None):
        """Log a Kumo RFM query execution"""
        try:
            query = """
                INSERT INTO kumo_rfm_queries 
                (use_case_id, query_text, query_type, execution_time_ms, result_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING query_id;
            """
            
            self.cursor.execute(query, (
                use_case_id,
                query_text,
                query_type,
                execution_time_ms,
                result_count,
                json.dumps(metadata) if metadata else None
            ))
            
            query_id = self.cursor.fetchone()[0]
            self.conn.commit()
            
            return query_id
            
        except psycopg2.Error as e:
            print(f"Error logging query: {e}")
            self.conn.rollback()
            return None
    
    def save_relationship(self, use_case_id: int, source: str, target: str, 
                         rel_type: str, strength: float = None, metadata: dict = None):
        """Save a discovered relationship"""
        try:
            query = """
                INSERT INTO kumo_rfm_relationships 
                (use_case_id, source_entity, target_entity, relationship_type, strength, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING relationship_id;
            """
            
            self.cursor.execute(query, (
                use_case_id,
                source,
                target,
                rel_type,
                strength,
                json.dumps(metadata) if metadata else None
            ))
            
            rel_id = self.cursor.fetchone()[0]
            self.conn.commit()
            
            return rel_id
            
        except psycopg2.Error as e:
            print(f"Error saving relationship: {e}")
            self.conn.rollback()
            return None
    
    def visualize_use_case_graph(self, use_case_id: int = None):
        """Visualize relationships from kumo_rfm_relationships table"""
        try:
            if use_case_id:
                query = """
                    SELECT source_entity, target_entity, relationship_type, strength
                    FROM kumo_rfm_relationships
                    WHERE use_case_id = %s
                    ORDER BY strength DESC NULLS LAST
                """
                self.cursor.execute(query, (use_case_id,))
            else:
                query = """
                    SELECT source_entity, target_entity, relationship_type, strength
                    FROM kumo_rfm_relationships
                    ORDER BY strength DESC NULLS LAST
                    LIMIT 100
                """
                self.cursor.execute(query)
            
            results = self.cursor.fetchall()
            
            if not results:
                print("No relationships found to visualize.")
                return
            
            # Create DataFrame and visualize
            df = pd.DataFrame(results, columns=['source', 'target', 'type', 'strength'])
            self._plot_graph(df)
            
        except psycopg2.Error as e:
            print(f"Error visualizing use case graph: {e}")
    
    def cleanup(self):
        """Close database connections"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print(f"\n{Fore.GREEN}✓ Disconnected from PostgreSQL{Style.RESET_ALL}")
    
    def run(self):
        """Main CLI loop - Direct SQL mode"""
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Kumo RFM CLI - PostgreSQL Database Management{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"Type {Fore.YELLOW}'help'{Style.RESET_ALL} for commands, {Fore.YELLOW}'viz'{Style.RESET_ALL} for visualizations, {Fore.YELLOW}'\\q'{Style.RESET_ALL} to quit")
        print()
        
        # Connect to database
        if not self.connect_postgres():
            print("Failed to connect to database. Exiting.")
            return
        
        print()
        
        # Go straight to SQL mode (no mode selection)
        try:
            self.run_sql_mode()
        except KeyboardInterrupt:
            print("\n\nExiting...")
        finally:
            self.cleanup()
    
    def run_sql_mode(self):
        """Unified SQL mode - like psql"""
        while True:
            try:
                query = input(f"{Fore.GREEN}sql>{Style.RESET_ALL} ").strip()
                
                if not query:
                    continue
                elif query in ['\\q', 'quit', 'exit']:
                    break
                elif query == 'help':
                    self.show_sql_help()
                elif query in ['viz', 'visualize']:
                    self.run_visualize_mode()
                elif query == '\\dt':
                    self.list_tables()
                elif query.startswith('\\d '):
                    table = query[3:].strip()
                    self.describe_table(table)
                elif query == '\\l':
                    self.list_databases()
                elif query.startswith('\\i '):
                    # Handle file import command
                    filename = query[3:].strip()
                    self.execute_sql_file(filename)
                else:
                    # Strip "sql>" prefix if user pasted it
                    query = self._clean_pasted_query(query)
                    self.last_query = query
                    self.execute_sql(query)
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use '\\q' to exit{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def _clean_pasted_query(self, query: str) -> str:
        """Clean pasted query by removing prompt prefixes like 'sql>'"""
        # Remove leading 'sql>' or similar prompts
        patterns = [
            r'^sql>\s*',
            r'^psql>\s*',
            r'^postgres=\#\s*',
            r'^postgres=-\s*'
        ]
        
        cleaned = query
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def execute_sql_file(self, filename: str):
        """Execute SQL commands from a file"""
        try:
            with open(filename, 'r') as f:
                sql_content = f.read()
            
            # Split by semicolons and execute each statement
            statements = [s.strip() for s in sql_content.split(';') if s.strip()]
            
            print(f"{Fore.CYAN}Executing {len(statements)} statements from {filename}...{Style.RESET_ALL}")
            
            for i, statement in enumerate(statements, 1):
                # Skip comments
                if statement.startswith('--'):
                    continue
                
                try:
                    print(f"\n{Fore.YELLOW}[{i}/{len(statements)}]{Style.RESET_ALL} {statement[:60]}...")
                    self.execute_sql(statement, store_results=False)
                except Exception as e:
                    print(f"{Fore.RED}Failed at statement {i}: {e}{Style.RESET_ALL}")
                    response = input(f"{Fore.YELLOW}Continue? (y/n): {Style.RESET_ALL}").strip().lower()
                    if response != 'y':
                        break
            
            print(f"\n{Fore.GREEN}✓ File execution complete{Style.RESET_ALL}")
            
        except FileNotFoundError:
            print(f"{Fore.RED}Error: File '{filename}' not found{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error reading file: {e}{Style.RESET_ALL}")
    
    def show_sql_help(self):
        """Show SQL mode help"""
        print(f"\n{Fore.CYAN}SQL Mode Commands:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}<any SQL>{Style.RESET_ALL}           - Execute SQL directly (CREATE TABLE, SELECT, INSERT, etc.)")
        print(f"  {Fore.GREEN}\\dt{Style.RESET_ALL}                 - List all tables")
        print(f"  {Fore.GREEN}\\d <table>{Style.RESET_ALL}          - Describe table structure")
        print(f"  {Fore.GREEN}\\l{Style.RESET_ALL}                  - List all databases")
        print(f"  {Fore.GREEN}\\i <file>{Style.RESET_ALL}           - Execute SQL from file")
        print(f"  {Fore.GREEN}viz{Style.RESET_ALL}                 - Switch to visualization mode")
        print(f"  {Fore.GREEN}help{Style.RESET_ALL}                - Show this help")
        print(f"  {Fore.GREEN}\\q, quit, exit{Style.RESET_ALL}      - Exit the CLI")
        print()
        print(f"{Fore.CYAN}Examples:{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}CREATE TABLE my_series (id TEXT PRIMARY KEY, category TEXT, is_favorite BOOLEAN);{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}INSERT INTO my_series VALUES ('FEDFUNDS', 'Inflation', true);{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}SELECT * FROM my_series WHERE is_favorite = true;{Style.RESET_ALL}")
        print()
        print(f"{Fore.MAGENTA}Tip: You can paste SQL with 'sql>' prefixes - they'll be automatically removed{Style.RESET_ALL}")
        print()


def main():
    cli = KumoRFMCLI()
    cli.run()


if __name__ == "__main__":
    main()
