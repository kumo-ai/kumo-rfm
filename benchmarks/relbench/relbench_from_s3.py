"""Script to run relbench experiments directly from the parquet files.

This script shows how to load the parquet files from an S3 bucket, along with a 
custom context table to build the graph, and make predictions.

The goal is to serve as an example that can be adapted to any other dataset,
not necessarily coming from relbench, by pointing to an S3 bucket with parquet
files, and by adding a custom context table to the graph.

Steps to run this:

STEP 0: [OPTIONAL] Generate context table
`python generate_context_table.py --dataset rel-avito --task ad-ctr`.
This is done following the relbench task definition for reproducibility,
but this script can be replaced with your own logic for generating the
context table, depending on the task.
Copy the table to the S3 bucket where the raw parquet files are stored.

STEP 1. Run predictions:
`python relbench_from_s3.py --s3_base_path s3://kumo-public-datasets/rel-bench/rel-avito/`
"""

import argparse

import boto3
import numpy as np
import pandas as pd
import tqdm
from kumoai.experimental import rfm
from sklearn.metrics import roc_auc_score


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
    is_regression: bool,
) -> rfm.LocalGraph:
    """Builds the graph from the datasets' parquet files."""
    df_dict = {
        name.replace('.parquet', ''): pd.read_parquet(f"{s3_path}{name}")
        for name in parquet_names
    }
    # Set the target to None for the test rows
    context_df = df_dict['context']
    context_df.loc[context_df['is_test'], 'TARGET'] = None
    df_dict['context'] = context_df

    graph = rfm.LocalGraph.from_data(
        df_dict,
        infer_metadata=True,
        verbose=False,
    )

    context_table = graph.tables["context"]
    context_table.primary_key = "index"
    context_table.time_column = "TIME"
    
    if is_regression:
        for column in context_table.columns:
            if column.name == "TARGET":
                column.stype = "numerical"
                break

    return graph


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Load parquet files from S3")
    parser.add_argument(
        "--s3_base_path",
        default="s3://kumo-public-datasets/rel-bench/rel-avito/",
        help="Base S3 path to load datasets from")
    parser.add_argument("--is_regression", action="store_true", default=True)
    parser.add_argument('--run_mode', type=str, default='best')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--max_test_steps', type=int, default=2)
    args = parser.parse_args()

    rfm.init()

    # ===============================================
    # STEP 1: BUILD THE GRAPH FROM THE S3 PATH
    # This will look for all parquet files in the s3 path and build the graph.
    # It expects to find a "context.parquet" file in the s3 path with the
    # custom context table.
    #
    # For the current script, the custom table looks like the following
    # dataframe example:
    # index  AdID           TIME    TARGET      is_test
    # 0     1047725     2015-05-14  0.021978     True
    # 1      383328     2015-05-14  0.032000     True
    # 2      383653     2015-05-14  0.005202     True
    # 3     4466016     2015-05-14  0.011364     True
    # ...      ...        ...       ...      ...
    # 3577   354386     2015-05-08  0.004491    False
    # 3578   360473     2015-05-08  0.090909    False
    # 3579   401949     2015-05-08  0.045455    False
    #
    # TODO: Set the desired s3 path pointing to the parquet files.
    # ===============================================
    print(f"S3 path: {args.s3_base_path}")
    parquet_names = list_parquet_files_from_s3(args.s3_base_path)
    print(f"Found parquet files: {parquet_names}")

    graph = get_graph(args.s3_base_path, parquet_names, args.is_regression)
    graph.print_metadata()
    graph.print_links()
    model = rfm.KumoRFM(graph)

    # ===============================================
    # STEP 2: DEFINE THE ENTITIES FOR WHICH TO MAKE PREDICTIONS
    # For the sake of this example, we load the context table again in order to
    # read the indices of the test rows, and also to read the targets for
    # evaluation, but any entities' ids could be defined here.
    # ===============================================
    context_df = pd.read_parquet(f"{args.s3_base_path}context.parquet")

    if args.is_regression:
        query = "PREDICT context.TARGET FOR context.index IN ({indices})"
    else:
        query = "PREDICT context.TARGET = 1 FOR context.index IN ({indices})"

    # ===============================================
    # STEP 3: PREDICT
    # This collects the ids defined in STEP 2 and makes predictions
    # ===============================================
    test_df = context_df[context_df['is_test']]
    test_df = test_df.sample(frac=1, random_state=24)
    test_df = test_df.reset_index(drop=True)

    ys_pred = []
    steps = list(range(0, len(test_df), args.batch_size))[:args.max_test_steps]
    for i, step in enumerate(tqdm.tqdm(steps)):
        indices = range(step, min(step + args.batch_size, len(test_df)))
        _query = query.format(indices=', '.join(str(i) for i in indices))
        
        # Runs the prediction
        df = model.predict(
            _query,
            run_mode=args.run_mode,
            anchor_time='entity',
            num_neighbors=[1, 16, 16],
            verbose=i == 0,
        )
        
        if args.is_regression:
            ys_pred.append(df['TARGET_PRED'].to_numpy())
        else:
            ys_pred.append(df['True_PROB'].to_numpy())

    # ===============================================
    # STEP 4: EVALUATE
    # This will evaluate the predictions by comparing them to the ground-truth
    # values.
    # For this example we take the values from the context table, but any other
    # source could be used.
    # TODO: Get the ground-truth values for the test rows.
    # ===============================================
    y_pred = np.concatenate(ys_pred)
    y_test = test_df["TARGET"].to_numpy()[:len(y_pred)]
    if args.is_regression:
        print(f'MAE: {np.abs(y_test - y_pred).mean():.4f}')
    else:
        print(f'AUROC: {roc_auc_score(y_test, y_pred):.4f}')
