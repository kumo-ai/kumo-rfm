"""Script to run various rel-bench tasks using PQL directly.

This script shows how to load the parquet files from an S3 bucket, build the
graph, and make predictions.

The goal is to serve as an example that can be adapted to any other dataset,
not necessarily coming from relbench, by pointing to an S3 bucket with parquet
files, and by evaluating a predictive query.
"""
import argparse

import numpy as np
import pandas as pd
from kumoai.experimental import rfm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--run_mode', type=str, default='best')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--max_test_steps', type=int, default=5)
args = parser.parse_args()

NUM_NEIGHBORS = {  # Optimal number of neighbors for each task:
    ('rel-amazon', 'item-ltv'): [64],
    ('rel-amazon', 'item-churn'): [128, 128],
    ('rel-amazon', 'user-ltv'): [16, 16],
    ('rel-amazon', 'user-churn'): [128, 128],
    ('rel-avito', 'ad-ctr'): [16, 16],
    ('rel-stack', 'user-engagement'): [8, 8],
}

ROOT = 's3://kumo-public-datasets/rel-bench'
FILENAMES: dict[str, list[str]] = {
    'rel-amazon': [
        'customer.parquet',
        'product.parquet',
        'review.parquet',
    ],
    'rel-avito': [
        'AdsInfo.parquet',
        'Category.parquet',
        'Location.parquet',
        'PhoneRequestsStream.parquet',
        'SearchInfo.parquet',
        'SearchStream.parquet',
        'UserInfo.parquet',
        'VisitStream.parquet',
    ],
    'rel-stack': [
        'badges.parquet',
        'comments.parquet',
        'postHistory.parquet',
        'postLinks.parquet',
        'posts.parquet',
        'users.parquet',
        'votes.parquet',
    ]
}

if (args.dataset, args.task) not in NUM_NEIGHBORS:
    raise NotImplementedError(f"'{args.task}' task on '{args.dataset}' not "
                              f"yet available")

rfm.init()

# Download the raw data files:
df_dict: dict[str, pd.DataFrame] = {}
for name in tqdm(FILENAMES[args.dataset], desc="Downloading data"):
    path = f'{ROOT}/{args.dataset}/{name}'
    df_dict[name.split('.')[0]] = pd.read_parquet(path)

# Pre-process data to model rel-bench tasks precisely:
if args.dataset == 'rel-avito':
    # Ensure that N/A values are assumed to be non-clicked:
    df_dict['SearchStream']['IsClick'].fillna(0.0, inplace=True)
elif args.dataset == 'rel-amazon' and args.task.endswith('ltv'):
    # Join 'price' to 'review' table to be able to reference it via PQL:
    df_dict['review'] = df_dict['review'].merge(
        df_dict['product'][['product_id', 'price']],
        on='product_id',
        how='left',
    )

graph = rfm.LocalGraph.from_data(df_dict)

if args.dataset == 'rel-avito':
    # Adjust the type of the 'IsClick' column to numerical:
    graph['SearchStream']['IsClick'].stype = 'numerical'

model = rfm.KumoRFM(graph)

# Define the predictive query:
if args.dataset == 'rel-amazon' and args.task == 'item-ltv':
    # The LTV (life-time value) for a product (the number of times a product is
    # reviewed in the next quarter multiplied by its price), given it was
    # reviewed in the last quarter.
    query = ("PREDICT SUM(review.price, 0, 91, days) "
             "FOR product.product_id IN ({indices}) "
             "ASSUMING COUNT(review.*, 0, 91, days)>0")
    anchor_time = pd.Timestamp('2016-01-01')

elif args.dataset == 'rel-amazon' and args.task == 'item-churn':
    # Predict if a product receives no review in the next quarter, given it was
    # reviewed at least once in the last quarter.
    query = ("PREDICT COUNT(review.*, 0, 91, days)=0 "
             "FOR product.product_id IN ({indices}) "
             "WHERE COUNT(review.*, -91, 0, days)>0")
    anchor_time = pd.Timestamp('2016-01-01')

elif args.dataset == 'rel-amazon' and args.task == 'user-ltv':
    # The LTV (life-time value) for a user (the number of times a user has
    # reviewed a product in the next quarter multiplied by its price), given
    # they have reviewed at least one product in the last quarter.
    query = ("PREDICT SUM(review.price, 0, 91, days) "
             "FOR customer.customer_id IN ({indices}) "
             "WHERE COUNT(review.*, -91, 0, days)>0")
    anchor_time = pd.Timestamp('2016-01-01')

elif args.dataset == 'rel-amazon' and args.task == 'user-churn':
    # Predict if a user does not review any product in the next quarter, given
    # they have reviewed at least one product in the last quarter.
    query = ("PREDICT COUNT(review.*, 0, 91, days)=0 "
             "FOR customer.customer_id IN ({indices}) "
             "WHERE COUNT(review.*, -91, 0, days)>0")
    anchor_time = pd.Timestamp('2016-01-01')

elif args.dataset == 'rel-avito' and args.task == 'ad-ctr':
    # Predict the average click-through rate (CTR) for the next 4 days for
    # each ad that has been clicked at least once.
    query = ("PREDICT AVG(SearchStream.IsClick, 0, 4, days) "
             "FOR AdsInfo.AdID IN ({indices}) "
             "ASSUMING SUM(SearchStream.IsClick, 0, 4, days)>0")
    anchor_time = pd.Timestamp('2015-05-14')

elif args.dataset == 'rel-stack' and args.task == 'user-engagement':
    # Predict if a user will make any votes/posts/comments in the next
    # quarter given they were previously active.
    query = ("PREDICT COUNT(votes.*, 0, 91, days)>0 OR "
             "        COUNT(posts.*, 0, 91, days)>0 OR "
             "        COUNT(comments.*, 0, 91, days)>0 "
             "FOR users.Id IN ({indices}) "
             "WHERE COUNT(votes.*, -INF, 0)>0 OR "
             "      COUNT(posts.*, -INF, 0)>0 OR "
             "      COUNT(comments.*, -INF, 0)>0")
    anchor_time = pd.Timestamp('2021-01-01')

# Define the entities for which to evaluate against:
# Here, we randomly sample entities (and obtain their corresponding
# ground-truth labels given by the PQ) at a specific anchor time.
test_df = model.get_train_table(
    query.format(indices='0, 1'),  # Dummy values.
    size=args.batch_size * args.max_test_steps,
    anchor_time=anchor_time,
    max_iterations=2000,  # Ensure that we find all rel-bench examples.
)

# Predict:
ys_pred = []
for i, step in enumerate(tqdm(range(0, len(test_df), args.batch_size))):
    test_entities = test_df['ENTITY'][step:step + args.batch_size]
    df = model.predict(
        query=query.format(indices=', '.join(str(i) for i in test_entities)),
        run_mode=args.run_mode,
        anchor_time=anchor_time,
        num_neighbors=NUM_NEIGHBORS[(args.dataset, args.task)],
        max_pq_iterations=1000,  # Ensure enough context labels are found.
        verbose=i == 0,  # Prevent excessive logging.
    )
    if 'True_PROB' in df:
        ys_pred.append(df['True_PROB'].to_numpy())
    else:
        ys_pred.append(df['TARGET_PRED'].to_numpy())

# Evaluate:
y_pred = np.concatenate(ys_pred)
if 'True_PROB' in df:
    y_true = test_df['TARGET'].to_numpy().astype(float)
    print(f'AUROC: {roc_auc_score(y_true, y_pred):.4f}')
else:
    y_true = test_df['TARGET'].to_numpy()
    print(f'MAE: {np.abs(y_true - y_pred).mean():.4f}')
