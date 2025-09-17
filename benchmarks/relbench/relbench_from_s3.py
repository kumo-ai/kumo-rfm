
import boto3
import pandas as pd
from io import BytesIO
import argparse
import numpy as np
import tqdm
from kumoai.experimental import rfm

TASKS = {  # All available regression tasks
    'rel-avito': ['ad-ctr'],
    'rel-event': ['user-attendance'],
    'rel-f1': ['driver-position'],
    'rel-hm': ['item-sales'],
    'rel-stack': ['post-votes'],
    'rel-trial': ['study-adverse', 'site-success'],
    'rel-amazon': ['user-ltv', 'item-ltv'],
}

NUM_NEIGHBORS = {  # Optimal number of neighbors for each task:
    ('rel-avito', 'ad-ctr'): [1, 16, 16],
    ('rel-event', 'user-attendance'): [1, 32, 32],
    ('rel-f1', 'driver-position'): [1, 8, 8],
    ('rel-hm', 'item-sales'): [1, 64],
    ('rel-stack', 'post-votes'): [1, 8, 8],
    ('rel-trial', 'study-adverse'): [1, 128, 128],
    ('rel-trial', 'site-success'): [1, 16, 16],
    ('rel-amazon', 'user-ltv'): [1, 16, 16],
    ('rel-amazon', 'item-ltv'): [1, 64],
}

rfm.init()

def load_parquet_files_from_s3(s3_path: str) -> dict[str, pd.DataFrame]:
    """
    Load all parquet files from an S3 path into a dictionary of pandas DataFrames.
    
    Args:
        s3_path (str): S3 path containing parquet files (e.g., "s3://bucket/path/")
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping filename to DataFrame
    """
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    if not s3_path.startswith('s3://'):
        raise ValueError("s3_path must start with 's3://'")
    
    s3_path = s3_path.rstrip('/')
    bucket_name = s3_path.split('/')[2]
    prefix = '/'.join(s3_path.split('/')[3:])
    
    if prefix:
        prefix += '/'
    
    # List all objects in the S3 path
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"No files found in {s3_path}")
        return {}
    
    parquet_files = {}
    
    for obj in response['Contents']:
        key = obj['Key']
        
        if key.endswith('.parquet'):
            filename = key.split('/')[-1]
            
            try:
                # Download the parquet file
                response_obj = s3_client.get_object(Bucket=bucket_name, Key=key)
                parquet_data = response_obj['Body'].read()
                
                # Read parquet data into DataFrame
                df = pd.read_parquet(BytesIO(parquet_data))
                parquet_files[filename] = df
                
                print(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue
    
    return parquet_files


def get_graph(s3_base_path: str, dataset: str) -> rfm.LocalGraph:
    dataframes = load_parquet_files_from_s3(f"{s3_base_path}{dataset}/")
    df_dict = {
        name.replace('.parquet', '').lower(): df 
        for name, df in dataframes.items()
    }
    graph = rfm.LocalGraph.from_data(df_dict, infer_metadata=True, verbose=False)
    return graph


# Example usage
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Load parquet files from S3")
    parser.add_argument("--dataset", default="rel-avito", 
                       help="Dataset name to load from S3 (default: rel-f1)")
    parser.add_argument(
        "--s3_base_path", 
        default="s3://kumo-public-datasets/rel-bench/", 
        help="Base S3 path to load datasets from")
    
    args = parser.parse_args()
    
    # Construct S3 path based on dataset argument
    s3_path = f"{args.s3_base_path}{args.dataset}/"
    
    print(f"Loading dataset: {args.dataset}")
    print(f"S3 path: {s3_path}")
    
    # Load all parquet files from the S3 path
    graph = get_graph(args.s3_base_path, args.dataset)
    graph.print_metadata()
    graph.print_links()
