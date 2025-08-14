import pandas as pd
from kumoai.experimental import rfm
import os,dotenv
from display import display_recommendations # Our online store

USER_CURRENT = 50  # This is me. You can set it to any user_id from 0 to 11,462.

root = "s3://rfm-shopping-recommender"
data = {
"users": pd.read_parquet(f"{root}/users.parquet"), 
"items": pd.read_parquet(f"{root}/items.parquet"),
"views": pd.read_parquet(f"{root}/views.parquet"),
"orders": pd.read_parquet(f"{root}/orders.parquet"),
"returns": pd.read_parquet(f"{root}/returns.parquet"),
}

dotenv.load_dotenv()
API_KEY = os.getenv("KUMO_API_KEY") # Get your own!

rfm.init(api_key=API_KEY)
graph = rfm.LocalGraph.from_data(data)
model = rfm.KumoRFM(graph)

query = (f"PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 10 FOR users.user_id={USER_CURRENT}")

pred_df = model.predict(query)
display_recommendations(pred_df) # Our own online store - create your own and try it out!
