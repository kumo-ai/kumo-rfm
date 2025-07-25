import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import kumoai as kumo
import kumoai.experimental.rfm as rfm
from kumoai.experimental.rfm import KumoRFM

# --- App Setup ---
st.set_page_config(page_title="Customer Support Assistant", layout="wide")
st.markdown("<h1 style='text-align: center;'>Customer Support Assistant</h1>", unsafe_allow_html=True)

# --- Load Data ---
# example tables from kumo public datasets
users_df = pd.read_parquet(f's3://kumo-sdk-public/rfm-datasets/online-shopping/users.parquet', storage_options={"anon": True})
items_df = pd.read_parquet(f's3://kumo-sdk-public/rfm-datasets/online-shopping/items.parquet', storage_options={"anon": True})
orders_df = pd.read_parquet(f's3://kumo-sdk-public/rfm-datasets/online-shopping/orders.parquet', storage_options={"anon": True})

# --- Load Environment Variables ---
load_dotenv()
KUMO_API_KEY = os.getenv("KUMO_API_KEY")

if not KUMO_API_KEY:
    raise ValueError("KUMO_API_KEY and API_URL must be set in environment variables")

# --- Kumo Init ---
rfm.init(
    api_key=KUMO_API_KEY,
)

users_table = rfm.LocalTable(df=users_df, name="users").infer_metadata()
items_table = rfm.LocalTable(df=items_df, name="items").infer_metadata()
orders_table = rfm.LocalTable(df=orders_df, name="orders").infer_metadata()

graph = rfm.LocalGraph(tables=[users_table, orders_table, items_table])
graph.link(src_table="orders", fkey="user_id", dst_table="users")
graph.link(src_table="orders", fkey="item_id", dst_table="items")

model = KumoRFM(graph)

# --- Helper Functions ---
# predict 90 day churn of a given customer
def churn_pred(customer_id):
    query = f"PREDICT COUNT(orders.*, 0, 90, days)=0 FOR users.user_id={customer_id}"
    prediction_result = model.predict(query)
    return prediction_result.True_PROB[0]

# calculate realized LTV for a given customer
def ltv_calc(customer_id):
    ltv = orders_df.loc[orders_df['user_id'] == customer_id, 'price'].sum()
    return ltv

# predict 90 day ltv of a given customer
def ltv_pred(customer_id):
    query = f"PREDICT SUM(orders.price, 0, 90, days) FOR users.user_id={customer_id}"
    prediction_result = model.predict(query)
    return prediction_result.TARGET_PRED[0]

# recommend top K items for a given customer
def rec_pred(customer_id, top_k=5):
    query = f"PREDICT LIST_DISTINCT(orders.item_id, 0, 90, days) RANK TOP {top_k} FOR users.user_id={customer_id}"
    prediction_result = model.predict(query)
    prediction_result = prediction_result.rename(columns={"CLASS": "item_id"}).merge(items_df, on="item_id", how="left").head(top_k)
    prediction_result = prediction_result.drop(columns=["ENTITY", "ANCHOR_TIMESTAMP"])
    return prediction_result

# find top N substitutes for a given item
def find_sub(item_id, top_n=5):
    category = items_df.loc[items_df['item_id'] == item_id, 'category'].values[0]
    category_items = items_df[items_df['category'] == category]
    item_popularity = orders_df[orders_df['item_id'].isin(category_items['item_id'])] \
        .groupby('item_id').size().reset_index(name='order_count')
    result = item_popularity.merge(items_df, on='item_id')
    return result.sort_values('order_count', ascending=False).head(top_n)

# --- Session state to persist customer info ---
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = None
    st.session_state.churn_prob = None
    st.session_state.realized_ltv = None
    st.session_state.predicted_ltv = None

# --- Customer ID Input ---
customer_id_input = st.text_input("Enter Customer ID:", key="cust_input")
st.markdown("""
    <style>
    div[data-testid="stTextInput"] {
        width: 300px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

if customer_id_input and st.session_state.customer_id is None:
    try:
        customer_id = int(customer_id_input)
        st.session_state.customer_id = customer_id
        st.session_state.churn_prob = churn_pred(customer_id)
        st.session_state.realized_ltv = ltv_calc(customer_id)
        st.session_state.predicted_ltv = ltv_pred(customer_id)
    except ValueError:
        st.error("Please enter a valid numeric Customer ID.")

# --- Show Dashboard If Customer ID is Set ---
if st.session_state.customer_id is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Churn Risk", value=f"{st.session_state.churn_prob:.2%}")
    col2.metric(label="Realized LTV", value=f"${st.session_state.realized_ltv:,.2f}")
    col3.metric(label="90-day Predicted LTV", value=f"${st.session_state.predicted_ltv:,.2f}")

    tab1, tab2 = st.tabs(["Item Substitution", "Personalized Recommendation"])

    with tab1:
        st.subheader("Find Item Substitutes")
        col_space, col_a, col_b, col_space2 = st.columns([1, 3, 2, 1])
        with col_a:
            item_id_input = st.text_input("Enter Item ID for Substitution:", key="sub_item")
        with col_b:
            with st.container():
                    st.markdown("""
                    <style>
                    div[data-testid='stNumberInput'] {
                        width: 150px;
                    }
                    </style>
                """, unsafe_allow_html=True)
                    top_n_sub = st.number_input("Top N Substitutes", min_value=1, max_value=20, value=5, key="sub_n")

        if item_id_input:
            try:
                item_id_input = int(item_id_input)
                sub_result = find_sub(item_id_input, top_n=int(top_n_sub))
                item_info = items_df[items_df['item_id'] == item_id_input]
                st.subheader("Item Details")
                st.dataframe(item_info)
                st.subheader("Top Substitutes")
                if len(sub_result) < int(top_n_sub):
                    st.warning(f"Only found {len(sub_result)} substitute(s) for the selected item.")
                st.dataframe(sub_result)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.subheader("Personalized Recommendations")
        with st.form("rec_form"):
            top_k = st.number_input("Top K Recommendations", min_value=1, max_value=20, value=5, key="rec_k")
            submitted = st.form_submit_button("Get Recommendations")

        if submitted:
            try:
                recs = rec_pred(st.session_state.customer_id, top_k)
                st.dataframe(recs)
            except Exception as e:
                st.error(f"Error: {e}")
