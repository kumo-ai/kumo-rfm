import argparse

import numpy as np
import pandas as pd
import tqdm
from kumoai.experimental import rfm

from relbench.datasets import get_dataset
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--run_mode', type=str, default='best')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--max_test_steps', type=int, default=5)
args = parser.parse_args()

rfm.init()

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

tasks: list[tuple[str, str]] = []
if args.dataset is None:
    for dataset, available_tasks in TASKS.items():
        for task in available_tasks:
            tasks.append((dataset, task))
elif args.task is None:
    for task in TASKS[args.dataset]:
        tasks.append((args.dataset, task))
else:
    tasks.append((args.dataset, args.task))


def get_graph(dataset: str) -> rfm.LocalGraph:
    db = get_dataset(dataset, download=True).get_db(upto_test_timestamp=False)
    df_dict = {
        table_name: table.df
        for table_name, table in db.table_dict.items()
    }

    graph = rfm.LocalGraph.from_data(df_dict, infer_metadata=False)

    for table_name, table in db.table_dict.items():  # Set graph metadata:
        graph[table_name].primary_key = table.pkey_col
        graph[table_name].time_column = table.time_col
        for fkey, dst_table in table.fkey_col_to_pkey_table.items():
            graph.link(table_name, fkey, dst_table)

    # Remove some entity features (for now) since they are high cardinality and
    # hinder proper generalization across in-context examples:
    if dataset == 'rel-hm':
        for column in graph['article'].columns:
            if column.name != 'article_id':
                del graph['article'][column.name]
    elif dataset == 'rel-amazon':
        for column in graph['product'].columns:
            if column.name not in {'product_id', 'price'}:
                del graph['product'][column.name]

    return graph


def add_context(
    graph: rfm.LocalGraph,
    dataset: str,
    task: str,
) -> rfm.LocalGraph:
    task = get_task(dataset, task, download=True)
    context_dfs = []
    for split in ['test', 'val']:
        df = task.get_table(split, mask_input_cols=False).df
        df = df.drop(columns='index', errors='ignore')
        df[task.target_col] = df[task.target_col].astype('float')
        if split == 'test':
            # Shuffle test set to avoid biased ordering:
            df = df.sample(frac=1, random_state=24).reset_index(drop=True)
            df[task.target_col] = None  # Do not leak test labels.
        context_dfs.append(df)
    context_df = pd.concat(context_dfs, axis=0, ignore_index=True)
    context_df = context_df.reset_index()
    context_df = context_df.rename(
        columns={
            task.entity_col: 'ENTITY',
            task.time_col: 'TIME',
            task.target_col: 'TARGET',
        })

    context_table = rfm.LocalTable(
        context_df,
        name='context',
        primary_key='index',
        time_column='TIME',
    )
    # Make sure that `TARGET` is of type "numerical" so that we treat the task
    # of regression downstream:
    context_table['TARGET'].stype = 'numerical'
    graph.add_table(context_table)
    graph.link(context_table.name, 'ENTITY', task.entity_table)

    return graph


for dataset_name, task_name in tasks:
    print("===============================================")
    print(f"Dataset: '{dataset_name}', Task: '{task_name}'")
    print("===============================================")
    graph = get_graph(dataset_name)
    graph = add_context(graph, dataset_name, task_name)
    model = rfm.KumoRFM(graph)

    query = "PREDICT context.TARGET FOR context.index IN ({indices})"

    ys_pred = []
    task = get_task(dataset_name, task_name, download=True)
    test_df = task.get_table('test', mask_input_cols=False).df
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
            num_neighbors=NUM_NEIGHBORS[(dataset_name, task_name)],
            verbose=i == 0,
        )
        ys_pred.append(df['TARGET_PRED'].to_numpy())

    y_pred = np.concatenate(ys_pred)
    y_test = test_df[task.target_col].to_numpy()[:len(y_pred)]
    print(f'MAE: {np.abs(y_test - y_pred).mean():.4f}')
