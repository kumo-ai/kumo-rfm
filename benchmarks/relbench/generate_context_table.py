import argparse

import pandas as pd
from relbench.tasks import get_task

CLS_TASKS = {  # All available classification tasks
    'rel-avito': ['user-visits', 'user-clicks'],
    'rel-event': ['user-repeat', 'user-ignore'],
    'rel-f1': ['driver-dnf', 'driver-top3'],
    'rel-hm': ['user-churn'],
    'rel-stack': ['user-engagement', 'user-badge'],
    'rel-trial': ['study-outcome'],
    'rel-amazon': ['user-churn', 'item-churn'],
}

REG_TASKS = {  # All available regression tasks
    'rel-avito': ['ad-ctr'],
    'rel-event': ['user-attendance'],
    'rel-f1': ['driver-position'],
    'rel-hm': ['item-sales'],
    'rel-stack': ['post-votes'],
    'rel-trial': ['study-adverse', 'site-success'],
    'rel-amazon': ['user-ltv', 'item-ltv'],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate context table")
    parser.add_argument("--dataset", default="rel-avito")
    parser.add_argument("--task", default="ad-ctr")
    parser.add_argument(
        "--splits",
        type=str,
        default="test,val",
        help="Comma-separated list of splits to include "
        "(e.g., 'test,val,train')",
    )
    args = parser.parse_args()

    if args.task in CLS_TASKS[args.dataset]:
        target_type = 'Int64'
    elif args.task in REG_TASKS[args.dataset]:
        target_type = 'float'
    else:
        raise ValueError(f"Task {args.task} is not supported")

    context_dfs = []
    for split in args.splits.split(','):
        task = get_task(args.dataset, args.task, download=True)
        df = task.get_table(split, mask_input_cols=False).df
        df = df.drop(columns='index', errors='ignore')
        # Set target type according to the task
        df[task.target_col] = df[task.target_col].astype(target_type)

        # Add boolean flag to indicate test rows
        df['is_test'] = (split == 'test')

        if split == 'test':
            # Shuffle test set to avoid biased ordering:
            df = df.sample(frac=1, random_state=24).reset_index(drop=True)
        context_dfs.append(df)

    context_df = pd.concat(context_dfs, axis=0, ignore_index=True)
    context_df = context_df.reset_index()
    context_df = context_df.rename(columns={
        task.time_col: 'TIME',
        task.target_col: 'TARGET',
    })

    print(context_df.head())
    context_file_name = "context_table.parquet"
    context_df.to_parquet(context_file_name, index=False)
    print(f"Saved context data to {context_file_name}")
