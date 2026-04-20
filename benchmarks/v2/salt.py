import argparse
import os

import huggingface_hub
import pandas as pd
from datasets import load_dataset
from kumoai.experimental import rfm
from kumoapi.typing import Stype

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--run_mode', type=str, default='best')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--max_test_steps', type=int, default=20)
args = parser.parse_args()

sale_tasks = [
    'SALESOFFICE',
    'SALESGROUP',
    'CUSTOMERPAYMENTTERMS',
    'SHIPPINGCONDITION',
    'HEADERINCOTERMSCLASSIFICATION',
]
item_tasks = [
    'PLANT',
    'SHIPPINGPOINT',
    'ITEMINCOTERMSCLASSIFICATION',
]

rfm.init()
huggingface_hub.login(os.environ['HUGGINGFACE_TOKEN'])


# Load the SALT dataset #######################################################
def get_df(table_name: str, split: str = 'train') -> pd.DataFrame:
    df = load_dataset('sap-ai-research/SALT', table_name, split=split)
    return df.to_pandas()


sales = pd.concat([
    get_df(table_name='salesdocuments', split='train'),
    get_df(table_name='salesdocuments', split='test'),
], axis=0, ignore_index=True)
items = pd.concat([
    get_df(table_name='salesdocument_items', split='train'),
    get_df(table_name='salesdocument_items', split='test'),
], axis=0, ignore_index=True)
customers = get_df(table_name='customers')
addresses = get_df(table_name='addresses')

# Sanitize data ###############################################################
# Merge timestamps into single `datetime` column:
date = sales['CREATIONDATE'].astype(str)
time = sales['CREATIONTIME'].astype(str)
sales['CREATIONDATETIME'] = pd.to_datetime(date + ' ' + time)
del sales['CREATIONDATE']
del sales['CREATIONTIME']
# Add timestamp to items:
items = pd.merge(
    left=items,
    right=sales[['SALESDOCUMENT', 'CREATIONDATETIME']],
    how='left',
    left_on='SALESDOCUMENT',
    right_on='SALESDOCUMENT',
)
# Remove auto-generated columns:
del sales['__index_level_0__']
del items['__index_level_0__']
del customers['__index_level_0__']
del addresses['__index_level_0__']
# Add missing primary key:
items['ID'] = range(len(items))
# Rename columns to align with task name:
sales = sales.rename(
    columns={'INCOTERMSCLASSIFICATION': 'HEADERINCOTERMSCLASSIFICATION'})
items = items.rename(
    columns={'INCOTERMSCLASSIFICATION': 'ITEMINCOTERMSCLASSIFICATION'})
# Drop unused target columns:
sales.drop(
    labels=[task for task in sale_tasks if task != args.task.upper()],
    axis=1,
    inplace=True,
)
items.drop(
    labels=[task for task in item_tasks if task != args.task.upper()],
    axis=1,
    inplace=True,
)
# Mask out test labels of the target column:
if args.task.upper() in sale_tasks:
    num_test = len(get_df(table_name='salesdocuments', split='test'))
    y_test = sales[args.task.upper()].iloc[-num_test:].to_numpy().copy()
    pkey_test = sales['SALESDOCUMENT'].iloc[-num_test:].to_numpy()
    task_pos = sales.columns.get_loc(args.task.upper())
    sales.iloc[-num_test:, task_pos] = None
    # join customers and addresses
    customers = pd.merge(
        left=customers,
        right=addresses,
        how='left',
        left_on='ADDRESSID',
        right_on='ADDRESSID',
    )
    customers.drop(columns=['ADDRESSID'], inplace=True)
elif args.task.upper() in item_tasks:
    num_test = len(get_df(table_name='salesdocument_items', split='test'))
    y_test = items[args.task.upper()].iloc[-num_test:].to_numpy().copy()
    pkey_test = items['ID'].iloc[-num_test:].to_numpy()
    task_pos = items.columns.get_loc(args.task.upper())
    items.iloc[-num_test:, task_pos] = None
else:
    raise ValueError(f"Unsupported task '{args.task}'")

# Create the graph ###########################################################
df_dict = {
    'sales': sales,
    'items': items,
    'customers': customers,
    'addresses': addresses,
}
if args.task.upper() in sale_tasks:
    df_dict.pop('addresses') # joined on customers table

graph = rfm.Graph.from_data(df_dict, infer_metadata=False, verbose=False)
graph['items']['PRODUCT'].stype = Stype.categorical
graph['sales'].primary_key = 'SALESDOCUMENT'
graph['sales'].time_column = 'CREATIONDATETIME'
graph['items'].primary_key = 'ID'
graph['items'].time_column = 'CREATIONDATETIME'
graph['customers'].primary_key = 'CUSTOMER'
graph.link(src_table='items', fkey='SALESDOCUMENT', dst_table='sales')
graph.link(src_table='items', fkey='SOLDTOPARTY', dst_table='customers')
graph.link(src_table='items', fkey='SHIPTOPARTY', dst_table='customers')
graph.link(src_table='items', fkey='PAYERPARTY', dst_table='customers')
graph.link(src_table='items', fkey='BILLTOPARTY', dst_table='customers')
if 'addresses' in df_dict:
    graph['addresses'].primary_key = 'ADDRESSID'
    graph.link(src_table='customers', fkey='ADDRESSID', dst_table='addresses')

graph.validate()
graph.print_metadata()
graph.print_links()

# Model Querying #############################################################
model = rfm.KumoRFM(graph)

if args.task.upper() in sale_tasks:
    num_neighbors = [64, 64, 8]
    query = (f"PREDICT sales.{args.task.upper()} "
             f"FOR EACH sales.SALESDOCUMENT")
else:
    num_neighbors = [32, 32, 8]  # reduce due to memory constraints
    query = (f"PREDICT items.{args.task.upper()} "
             f"FOR EACH items.ID")

with model.batch_mode(batch_size=args.batch_size):
    indices = pkey_test[:args.max_test_steps * args.batch_size].tolist()
    df = model.predict(
        query=query,
        indices=indices,
        run_mode=args.run_mode,
        anchor_time='entity',  # NOTE Use entity table time as anchor time.
        num_neighbors=num_neighbors,
        inference_config={
            'num_estimators': 2,
            'column_shuffle': True,
            'category_shuffle': True,
            'class_shuffle': True,
            'hop_shuffle': True,
        },
    )
    y_pred = df['CLASS'].to_numpy().reshape(len(indices), -1)

# Metric Computation ##########################################################
match = y_test[:len(y_pred)].reshape(-1, 1) == y_pred
rank = match.astype(float).argmax(axis=-1) + 1
reciprocal_rank = 1.0 / rank
reciprocal_rank[match.sum(axis=-1) == 0.0] = 0.0
print(f'MRR: {reciprocal_rank.mean():.4f}')
