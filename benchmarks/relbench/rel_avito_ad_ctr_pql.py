"""Script to run rel-avito ad ctr task using PQL.

This script shows how to load the parquet files from an S3 bucket, build the
graph, and make predictions.

The goal is to serve as an example that can be adapted to any other dataset,
not necessarily coming from relbench, by pointing to an S3 bucket with parquet
files, and by writing the predictive query.

Steps to run this:
python rel_avito_ad_ctr_pql.py --s3_base_path s3://path/to/parquet/files/

Ex:
python rel_avito_ad_ctr_pql.py --s3_base_path s3://kumo-public-datasets/rel-bench/rel-avito/
"""

import argparse

import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm
from kumoai.experimental import rfm


def list_parquet_files_from_s3(s3_path: str) -> list[str]:
    """List all parquet files from an S3 path.

    Args:
        s3_path (str): S3 path containing parquet files

    Returns:
        List[str]: List of parquet filenames
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
        return []

    parquet_files = []

    for obj in response['Contents']:
        key = obj['Key']

        if key.endswith('.parquet'):
            filename = key.split('/')[-1]
            parquet_files.append(filename)

    return parquet_files


def get_graph(
    s3_path: str,
    parquet_names: list[str],
) -> rfm.LocalGraph:
    """Builds the graph from the datasets' parquet files."""
    df_dict = {
        name.replace('.parquet', ''): pd.read_parquet(f"{s3_path}{name}")
        for name in parquet_names
    }

    graph = rfm.LocalGraph.from_data(
        df_dict,
        infer_metadata=True,
        verbose=False,
    )

    return graph


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Load parquet files from S3")
    parser.add_argument(
        "--s3_base_path",
        default="s3://kumo-public-datasets/rel-bench/rel-avito/",
        help="Base S3 path to load datasets from")
    parser.add_argument('--run_mode', type=str, default='best')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--max_test_steps', type=int, default=20)
    args = parser.parse_args()

    rfm.init()

    # ===============================================
    # STEP 1: BUILD THE GRAPH FROM THE S3 PATH
    # This will look for all parquet files in the s3 path and build the graph.
    #
    # TODO: Set the desired s3 path pointing to the parquet files.
    # ===============================================
    print(f"S3 path: {args.s3_base_path}")
    parquet_names = list_parquet_files_from_s3(args.s3_base_path)
    parquet_names.remove("context.parquet")
    print(f"Found parquet files: {parquet_names}")

    graph = get_graph(args.s3_base_path, parquet_names)
    # Set the type of the IsClick column to numerical for the regression task
    graph['SearchStream']['IsClick'].stype = 'numerical'
    graph.print_metadata()
    graph.print_links()

    model = rfm.KumoRFM(graph)

    # ===============================================
    # STEP 2: DEFINE THE PREDICTIVE QUERY
    # For rel-avito ad-ctr, we predict the average click-through rate (CTR)
    # for the next 4 days for each ad that has been clicked at least once
    # ===============================================
    query = ("PREDICT AVG(SearchStream.IsClick, 0, 4, days) "
             "FOR AdsInfo.AdID IN ({indices}) "
             "WHERE SUM(SearchStream.IsClick, -INF, 0, days) > 0")

    # ===============================================
    # STEP 3: DEFINE THE ENTITIES FOR WHICH TO MAKE PREDICTIONS
    # We use the KumoRFM.get_train_table(...) helper function to get the 
    # training table corresponding to this PQL
    # ===============================================
    anchor_time = pd.Timestamp('2015-05-16')
    test_df = model.get_train_table(
        query.format(indices='0, 1'),  # Dummy values.
        size=args.batch_size * args.max_test_steps,
        anchor_time=anchor_time,
        max_iterations=1000,
    )
    test_indices = test_df["ENTITY"].tolist()

    # ===============================================
    # STEP 4: PREDICT
    # ===============================================
    ys_pred = []
    for i, step in enumerate(tqdm(range(0, len(test_df), args.batch_size))):
        indices = test_indices[step:step + args.batch_size]
        _query = query.format(indices=', '.join(str(i) for i in indices))
        df = model.predict(
            _query,
            run_mode=args.run_mode,
            anchor_time=anchor_time,
            max_pq_iterations=1000,  # Ensure enough context labels are found.
            verbose=i == 0,  # Prevent excessive logging.
        )
        ys_pred.append(df['TARGET_PRED'].to_numpy())

    y_pred = np.concatenate(ys_pred)
    y_test = test_df["TARGET"].to_numpy()

    # ===============================================
    # STEP 5: EVALUATE
    # This will evaluate the predictions by comparing them to the ground-truth
    # values.
    # For this example we take the values from the training table, but any 
    # other source could be used.
    # ===============================================
    print(f'MAE: {np.abs(y_test - y_pred).mean():.4f}')
