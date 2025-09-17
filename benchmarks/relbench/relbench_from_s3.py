import argparse
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
import tqdm
from kumoai.experimental import rfm
from sklearn.metrics import roc_auc_score

REG_TASKS = {  # All available regression tasks
    'rel-avito': ['ad-ctr'],
    'rel-event': ['user-attendance'],
    'rel-f1': ['driver-position'],
    'rel-hm': ['item-sales'],
    'rel-stack': ['post-votes'],
    'rel-trial': ['study-adverse', 'site-success'],
    'rel-amazon': ['user-ltv', 'item-ltv'],
}

ENTITY_TABLES = {
    # Regression tasks
    ('rel-avito', 'ad-ctr'): 'AdsInfo',
    ('rel-event', 'user-attendance'): 'users',
    ('rel-f1', 'driver-position'): 'drivers',
    ('rel-hm', 'item-sales'): 'article',
    ('rel-stack', 'post-votes'): 'posts',
    ('rel-trial', 'study-adverse'): 'studies',
    ('rel-trial', 'site-success'): 'facilities',
    ('rel-amazon', 'user-ltv'): 'customer',
    ('rel-amazon', 'item-ltv'): 'product',
    # Classification tasks
    ('rel-avito', 'user-visits'): 'UserInfo',
    ('rel-avito', 'user-clicks'): 'UserInfo',
    ('rel-event', 'user-repeat'): 'users',
    ('rel-event', 'user-ignore'): 'users',
    ('rel-f1', 'driver-dnf'): 'drivers',
    ('rel-f1', 'driver-top3'): 'drivers',
    ('rel-hm', 'user-churn'): 'customer',
    ('rel-stack', 'user-engagement'): 'users',
    ('rel-stack', 'user-badge'): 'users',
    ('rel-trial', 'study-outcome'): 'studies',
    ('rel-amazon', 'user-churn'): 'customer',
    ('rel-amazon', 'item-churn'): 'product',
}

NUM_NEIGHBORS = {  # Optimal number of neighbors for each task:
    # Regression tasks
    ('rel-avito', 'ad-ctr'): [1, 16, 16],
    ('rel-event', 'user-attendance'): [1, 32, 32],
    ('rel-f1', 'driver-position'): [1, 8, 8],
    ('rel-hm', 'item-sales'): [1, 64],
    ('rel-stack', 'post-votes'): [1, 8, 8],
    ('rel-trial', 'study-adverse'): [1, 128, 128],
    ('rel-trial', 'site-success'): [1, 16, 16],
    ('rel-amazon', 'user-ltv'): [1, 16, 16],
    ('rel-amazon', 'item-ltv'): [1, 64],
    # Classification tasks
    ('rel-avito', 'user-clicks'): [1, 128, 128],
    ('rel-avito', 'user-visits'): [1, 128, 128],
    ('rel-f1', 'driver-dnf'): [1, 16, 16],
    ('rel-f1', 'driver-top3'): [1, 8, 8],
    ('rel-hm', 'user-churn'): [1, 64, 64],
    ('rel-event', 'user-repeat'): [1, 64, 64],
    ('rel-event', 'user-ignore'): [1, 16, 16],
    ('rel-stack', 'user-engagement'): [1, 8, 8],
    ('rel-stack', 'user-badge'): [1, 8, 8],
    ('rel-trial', 'study-outcome'): [1, 64, 64],
    ('rel-amazon', 'user-churn'): [1, 128, 128],
    ('rel-amazon', 'item-churn'): [1, 128, 128],
}


def load_parquet_files_from_s3(s3_path: str) -> dict[str, pd.DataFrame]:
    """Load all parquet files from an S3 path into a dictionary of pandas
    DataFrames.

    Args:
        s3_path (str): S3 path containing parquet files

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
                response_obj = s3_client.get_object(Bucket=bucket_name,
                                                    Key=key)
                parquet_data = response_obj['Body'].read()

                # Read parquet data into DataFrame
                df = pd.read_parquet(BytesIO(parquet_data))
                parquet_files[filename] = df

                print(f"Loaded {filename}: "
                      f"{df.shape[0]} rows, {df.shape[1]} columns")

            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue

    return parquet_files


def get_graph(dataframes: dict[str, pd.DataFrame]) -> rfm.LocalGraph:
    """Builds the graph from the datasets' dataframes without including the
    context table.
    """
    df_dict = {
        name.replace('.parquet', ''): df
        for name, df in dataframes.items()
        if not name.startswith('context_table_')
    }
    graph = rfm.LocalGraph.from_data(
        df_dict,
        infer_metadata=True,
        verbose=False,
    )
    return graph


def get_context_df(
    task: str,
    dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Load the context table from the dataframes."""
    context_file_name = f"context_table_{task.replace('-', '_')}.parquet"
    try:
        context_df = dataframes[context_file_name]
    except KeyError:
        raise KeyError(
            f"Context table {context_file_name} not found in dataframes. "
            f"Make sure to create and upload the context table to the S3 path "
            f"before running this script.") from None
    return context_df


def add_context(
    graph: rfm.LocalGraph,
    dataset: str,
    task: str,
    context_df: pd.DataFrame,
    is_regression: bool,
) -> rfm.LocalGraph:
    """Load and add context table to the graph removing the target values for
    the test rows so there is no leakage.

    Args:
        graph: The graph to add the context table to
        dataset: Dataset name (e.g., 'rel-avito')
        task: Task name (e.g., 'ad-ctr')
        context_df: Context DataFrame
        is_regression: Whether the task is a regression task
    """
    # Set TARGET to None for all rows where is_test is True,
    # without modifying the original df
    context_df = context_df.copy()
    context_df.loc[context_df['is_test'], 'TARGET'] = None

    context_table = rfm.LocalTable(
        context_df,
        name='context',
        primary_key='index',
        time_column='TIME',
    )
    if is_regression:
        # Make sure that `TARGET` is of type "numerical" so that we treat the
        # task of regression downstream:
        context_table['TARGET'].stype = 'numerical'

    graph.add_table(context_table)
    entity_table_name = ENTITY_TABLES[(dataset, task)]
    graph.link(context_table.name, 'ENTITY', entity_table_name)
    return graph


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Load parquet files from S3")
    parser.add_argument("--s3_base_path",
                        default="s3://kumo-public-datasets/rel-bench/",
                        help="Base S3 path to load datasets from")
    parser.add_argument("--dataset", default="rel-avito")
    parser.add_argument("--task", default="ad-ctr")
    parser.add_argument('--run_mode', type=str, default='best')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--max_test_steps', type=int, default=2)
    args = parser.parse_args()

    rfm.init()

    print("===============================================")
    print(f"Dataset: '{args.dataset}', Task: '{args.task}'")
    print("===============================================")

    is_regression = args.task in REG_TASKS[args.dataset]

    # ===============================================
    # STEP 1: LOAD PARQUET FILES FROM S3
    # This will look for all parquet files in the s3 path and load them into a
    # dictionary of dataframes.
    # TODO: Set the desired s3 path
    # ===============================================
    s3_path = f"{args.s3_base_path}{args.dataset}/"
    print(f"Loading dataset: {args.dataset}")
    print(f"S3 path: {s3_path}")
    dataframes = load_parquet_files_from_s3(s3_path)

    # Builds the graph with the original dataset's tables (without context)
    graph = get_graph(dataframes)


    # ===============================================
    # STEP 2: LOAD CONTEXT TABLE AND ADD IT TO THE GRAPH
    # This will load the context table from the dataframes and add it to the
    # graph.
    # In case of using a custom context table, create it and add it to the
    # graph, without the target values for the test rows so there is no leakage
    # TODO: Set the desired custom context table
    # ===============================================
    context_df = get_context_df(args.task, dataframes)
    graph = add_context(graph, args.dataset, args.task, context_df,
                        is_regression)
    graph.print_metadata()
    graph.print_links()

    model = rfm.KumoRFM(graph)

    if is_regression:
        query = "PREDICT context.TARGET FOR context.index IN ({indices})"
    else:
        query = "PREDICT context.TARGET = 1 FOR context.index IN ({indices})"

    # ===============================================
    # STEP 3: PREDICT
    # This collects the indices of the test rows and predicts the target values
    # for them.
    # For teh current script, the custom table looks like the following 
    # dataframe example:
    # index  ENTITY           TIME    TARGET  is_test
    # 0     1047725     2015-05-14  0.021978     True
    # 1      383328     2015-05-14  0.032000     True
    # 2      383653     2015-05-14  0.005202     True
    # 3     4466016     2015-05-14  0.011364     True
    # ...      ...        ...       ...      ...
    # 3577   354386     2015-05-08  0.004491    False
    # 3578   360473     2015-05-08  0.090909    False
    # 3579   401949     2015-05-08  0.045455    False
    # 
    # TODO: Based on the custom context table, collect the indices of the test
    # rows and predict the target values for them.
    # ===============================================
    ys_pred = []
    test_df = context_df[context_df['is_test']]
    test_df = test_df.sample(frac=1, random_state=24)
    test_df = test_df.reset_index(drop=True)
    steps = list(range(0, len(test_df), args.batch_size))[:args.max_test_steps]
    for i, step in enumerate(tqdm.tqdm(steps)):
        indices = range(step, min(step + args.batch_size, len(test_df)))
        _query = query.format(indices=', '.join(str(i) for i in indices))
        df = model.predict(
            _query,
            run_mode=args.run_mode,
            anchor_time='entity',
            num_neighbors=NUM_NEIGHBORS[(args.dataset, args.task)],
            verbose=i == 0,
        )
        if is_regression:
            ys_pred.append(df['TARGET_PRED'].to_numpy())
        else:
            ys_pred.append(df['True_PROB'].to_numpy())

    # ===============================================
    # STEP 4: EVALUATE
    # This will evaluate the predictions by comparing them to the ground-truth
    # values.
    # TODO: Based on the custom context table, collect the ground-truth values
    # for the test rows.
    # ===============================================
    y_pred = np.concatenate(ys_pred)
    y_test = test_df["TARGET"].to_numpy()[:len(y_pred)]
    if is_regression:
        print(f'MAE: {np.abs(y_test - y_pred).mean():.4f}')
    else:
        print(f'AUROC: {roc_auc_score(y_test, y_pred):.4f}')
