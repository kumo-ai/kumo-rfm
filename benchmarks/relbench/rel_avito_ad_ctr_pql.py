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
import tqdm
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
    parser.add_argument('--run_mode', type=str, default='fast')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--max_test_steps', type=int, default=2)
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
    # STEP 2: DEFINE THE ENTITIES FOR WHICH TO MAKE PREDICTIONS
    # For the sake of this example, we load the context table in order to
    # read the indices of the test rows, and also to read the targets for
    # evaluation, but any entities' ids could be defined here.
    # ===============================================
    context_df = pd.read_parquet(f"{args.s3_base_path}context.parquet")
    test_df = context_df[context_df['is_test']]
    test_df = test_df.drop(columns=["index", "is_test"], errors="ignore")

    entity_col = "AdID"
    target_col = "TARGET"
    nested_test_df = test_df.groupby("TIME")[[
        entity_col,
        target_col,
    ]].agg(list)

    # Since rel-bench datasets may contain multiple anchor timestamps, we first
    # group entities by their anchor timestamp, and then split entities within
    # the same anchor timestamp into chunks of size batch_size. This ensures
    # that for a single model.predict(...) call, we use the same anchor time in
    # order to be able to share the context among all entities to predict for
    test_indices = []
    for anchor_time, row in nested_test_df.iterrows():
        for step in range(0, len(row[entity_col]), args.batch_size):
            test_indices.append((
                anchor_time,
                row[entity_col][step:step + args.batch_size],
                row[target_col][step:step + args.batch_size],
            ))
        if len(test_indices) >= args.max_test_steps:
            break

    # Limit the number of test steps:
    test_indices = test_indices[:args.max_test_steps]

    # ===============================================
    # STEP 3: PREDICT
    # ===============================================
    ys_test = []
    ys_pred = []
    for i, (anchor_time, indices, y_test) in enumerate(tqdm.tqdm(test_indices)):
        _query = query.format(indices=', '.join(str(i) for i in indices))
        df = model.predict(
            _query,
            run_mode=args.run_mode,
            anchor_time=anchor_time,
            max_pq_iterations=1000,  # Ensure enough context labels are found.
            verbose=i == 0,  # Prevent excessive logging.
        )
        ys_pred.append(df['TARGET_PRED'].to_numpy())
        ys_test.append(np.array(y_test))

    y_pred = np.concatenate(ys_pred)
    y_test = np.concatenate(ys_test)

    # ===============================================
    # STEP 4: EVALUATE
    # This will evaluate the predictions by comparing them to the ground-truth
    # values.
    # For this example we take the values from the context table, but any other
    # source could be used.
    # TODO: Get the ground-truth values for the test rows.
    # ===============================================
    print(f'MAE: {np.abs(y_test - y_pred).mean():.4f}')
