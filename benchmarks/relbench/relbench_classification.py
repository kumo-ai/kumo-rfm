import argparse
import os

import numpy as np
import pandas as pd
import tqdm
from kumoai.experimental import rfm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from relbench.datasets import get_dataset
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--run_mode', type=str, default='best')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--max_test_steps', type=int, default=10)
args = parser.parse_args()

if 'KUMO_API_KEY' not in os.environ:
    rfm.authenticate()

rfm.init()

TASKS = {  # All available classification tasks
    'rel-avito': ['user-visits', 'user-clicks'],
    'rel-event': ['user-repeat', 'user-ignore'],
    'rel-f1': ['driver-dnf', 'driver-top3'],
    'rel-hm': ['user-churn'],
    'rel-stack': ['user-engagement', 'user-badge'],
    'rel-trial': ['study-outcome'],
    'rel-amazon': ['user-churn', 'item-churn'],
}

NUM_NEIGHBORS = {  # Optimal number of neighbors for each task:
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

    return graph


def add_context(
    graph: rfm.LocalGraph,
    dataset: str,
    task: str,
) -> rfm.LocalGraph:
    task = get_task(dataset, task, download=True)
    context_dfs = []
    for split in ['test', 'val', 'train']:
        df = task.get_table(split, mask_input_cols=False).df
        df = df.drop(columns='index', errors='ignore')
        df[task.target_col] = df[task.target_col].astype('Int64')
        if split == 'test':
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

    query = "PREDICT context.TARGET = 1 FOR context.index IN ({indices})"

    ys_pred = []
    task = get_task(dataset_name, task_name, download=True)
    test_df = task.get_table('test', mask_input_cols=False).df
    test_df[task.target_col] = test_df[task.target_col].astype(int)
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
        ys_pred.append(df['True_PROB'].to_numpy())

    y_pred = np.concatenate(ys_pred)
    y_test = test_df[task.target_col].to_numpy()[:len(y_pred)]
    
    # Convert probabilities to binary predictions using 0.5 threshold
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    auroc = roc_auc_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred_binary, average='macro')
    f1_micro = f1_score(y_test, y_pred_binary, average='micro')
    
    print(f'AUROC: {auroc:.4f}')
    print(f'AUPRC: {auprc:.4f}')
    print(f'F1 Macro: {f1_macro:.4f}')
    print(f'F1 Micro: {f1_micro:.4f}')
