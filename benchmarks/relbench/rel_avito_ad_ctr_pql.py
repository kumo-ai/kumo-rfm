"""Script to run rel-avito ad ctr task using PQL.

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
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str, default='best')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--max_test_steps', type=int, default=5)
args = parser.parse_args()

root = 's3://kumo-public-datasets/rel-bench/rel-avito'
filenames = [
    'AdsInfo.parquet',
    'Category.parquet',
    'Location.parquet',
    'PhoneRequestsStream.parquet',
    'SearchInfo.parquet',
    'SearchStream.parquet',
    'UserInfo.parquet',
    'VisitStream.parquet',
]

rfm.init()

# Step 1: Build the graph from a set of table files:
df_dict: dict[str, pd.DataFrame] = {}
for name in tqdm(filenames, desc="Downloading data"):
    df_dict[name.split('.')[0]] = pd.read_parquet(f'{root}/{name}')

# Ensure that N/A values are assumed to be non-clicked:
df_dict['SearchStream']['IsClick'].fillna(0.0, inplace=True)

graph = rfm.LocalGraph.from_data(df_dict)

# Set the type of the 'IsClick' column to numerical for the regression task:
graph['SearchStream']['IsClick'].stype = 'numerical'

model = rfm.KumoRFM(graph)

# Step 2: Define the predictive query
# For rel-avito ad-ctr, we predict the average click-through rate (CTR)
# for the next 4 days for each ad that has been clicked at least once.
query = ("PREDICT AVG(SearchStream.IsClick, 0, 4, days) "
         "FOR AdsInfo.AdID IN ({indices}) "
         "ASSUMING SUM(SearchStream.IsClick, 0, 4, days)>0")

# Step 3: Define the entities for which to evaluate against
# Here, we randomly sample entities (and obtain their corresponding
# ground-truth labels given by the PQ) at a specific anchor time.
anchor_time = pd.Timestamp('2015-05-14')
test_df = model.get_train_table(
    query.format(indices='0, 1'),  # Dummy values.
    size=args.batch_size * args.max_test_steps,
    anchor_time=anchor_time,
    max_iterations=2000,  # Ensure that we find all rel-bench examples.
)

# Step 4: Predict
ys_pred = []
for i, step in enumerate(tqdm(range(0, len(test_df), args.batch_size))):
    test_entities = test_df['ENTITY'][step:step + args.batch_size]
    df = model.predict(
        query=query.format(indices=', '.join(str(i) for i in test_entities)),
        run_mode=args.run_mode,
        anchor_time=anchor_time,
        num_neighbors=[16, 16],
        max_pq_iterations=1000,  # Ensure enough context labels are found.
        verbose=i == 0,  # Prevent excessive logging.
    )
    ys_pred.append(df['TARGET_PRED'].to_numpy())

y_pred = np.concatenate(ys_pred)

# Step 5: Evaluate
# Evaluate the predictions by comparing them to the ground-truth labels.
print(f'MAE: {np.abs(test_df["TARGET"].to_numpy() - y_pred).mean():.4f}')
