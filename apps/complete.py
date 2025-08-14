import copy

import pandas as pd
import streamlit as st
from kumoai.experimental import rfm
from streamlit_option_menu import option_menu
import dotenv, os
USER_ID = 50  # This is me.


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "users": pd.read_parquet("users.parquet"),
        "items": pd.read_parquet("items.parquet"),
        "views": pd.read_parquet("views.parquet"),
        "orders": pd.read_parquet("orders.parquet"),
        "returns": pd.read_parquet("returns.parquet"),
    }


@st.cache_resource
def load_model(df_dict: pd.DataFrame) -> rfm.KumoRFM:
    dotenv.load_dotenv()
    rfm.init(api_key = os.getenv("KUMO_API_KEY"))
    graph = rfm.LocalGraph.from_data(df_dict)
    return rfm.KumoRFM(graph)


if "data" not in st.session_state:
    st.session_state.data = copy.deepcopy(load_data())


def display_items(df: pd.DataFrame, title: str, button: str) -> None:
    if len(df) == 0:
        return
    st.markdown(f"**{title}**")
    cols = st.columns(3)
    for i, row in enumerate(df.itertuples()):
        with cols[i % 3]:
            st.image(row.image_url)
            st.markdown(f'**{row.prod_name}**')
            st.button(f"🛒 {button}", key=f"{button}_{i}")


st.set_page_config(page_title="Fashion Shop", layout="wide")
st.title("Fashion Shop 👗")

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "---",
            "Dress",
            "Trousers",
            "Sweater",
            "T-shirt",
            "---",
            "Past Orders",
            "Returns",
        ],
        icons=[
            "house",
            "",
            "1-square",
            "2-square",
            "3-square",
            "4-square",
            "",
            "archive",
            "arrow-return-left",
        ],
        default_index=0,
    )

if selected == "Home":
    st.subheader("Hi Matthias, welcome back!")

    if "churn" not in st.session_state:
        model = load_model(load_data())
        query = (f"PREDICT COUNT(orders.*, 0, 30, days)=0 "
                 f"FOR users.user_id={USER_ID}")
        st.session_state.churn = model.predict(query)

    if "churn" in st.session_state:
        if st.session_state.churn['True_PROB'][0] > 0.7:
            st.success("A special thank you - click here to receive your "
                       "$10 coupon.")

    if "rec_items" not in st.session_state:
        model = load_model(load_data())
        query = (f"PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) "
                 f"RANK TOP 10"
                 f"FOR users.user_id={USER_ID}")
        pred_df = model.predict(query)

        items_df = pd.merge(
            st.session_state.data['items'],
            pred_df[['CLASS']],
            how='right',
            left_on='item_id',
            right_on='CLASS',
        )

        # Find repeated purchases:
        orders_df = st.session_state.data['orders']
        orders_df = orders_df[orders_df['user_id'] == USER_ID]
        mask = items_df['item_id'].isin(orders_df['item_id'])

        st.session_state.rec_items = items_df[~mask][:3]
        st.session_state.repeated_items = items_df[mask][:3]

    if 'rec_items' in st.session_state:
        display_items(
            df=st.session_state.rec_items,
            title="Here are a few things you might like:",
            button="Order Now",
        )

    if 'repeated_items' in st.session_state:
        display_items(
            df=st.session_state.repeated_items,
            title="Loved it once? Love it again:",
            button="Buy Again",
        )

elif selected == "Returns":
    st.subheader("Returns")

    returns_df = st.session_state.data['returns']
    returns_df = returns_df[returns_df['user_id'] == USER_ID]
    returns_df = returns_df.sort_values('date', ascending=False)
    items_df = st.session_state.data['items']

    if 'reason' not in st.session_state:
        model = load_model(load_data())
        return_ids = returns_df['return_id'].tolist()
        query = (f"PREDICT returns.reason for returns.return_id IN "
                 f"({', '.join(str(x) for x in return_ids)})")
        st.session_state.reason = model.predict(query)

    # Visualize items returned items:
    for i, row in enumerate(returns_df.itertuples(index=False)):
        item = items_df.iloc[row.item_id]
        image_col, msg_col = st.columns([1, 4])
        with image_col:
            st.image(item['image_url'], use_container_width=True)
        with msg_col:
            st.markdown(f"**{item['prod_name']}**")
            st.markdown(f"**{row.date.date()}**")
            st.markdown(f"**${item['price']:.2f}**")
            if 'reason' in st.session_state:
                options = st.session_state.reason['CLASS'].tolist()
                options = options[5 * i:5 * (i + 1)]
                st.selectbox("Reason:", options, key=f"returns_{i}")

        if i + 1 != len(returns_df):
            st.markdown("---")

elif selected == "Past Orders":
    st.subheader("Past Orders")

    orders_df = st.session_state.data['orders']
    orders_df = orders_df[orders_df['user_id'] == USER_ID]
    orders_df = orders_df.sort_values('date', ascending=False)
    items_df = st.session_state.data['items']

    # Visualize all past orders:
    for i, row in enumerate(orders_df.itertuples(index=False)):
        item = items_df.iloc[row.item_id]
        image_col, msg_col = st.columns([1, 4])
        with image_col:
            st.image(item['image_url'], use_container_width=True)
        with msg_col:
            st.markdown(f"**{item['prod_name']}**")
            st.markdown(f"**{row.date.date()}**")
            st.markdown(f"**${item['price']:.2f}**")
            st.caption(item['detail_desc'])
        if i + 1 != len(orders_df):
            st.markdown("---")

else:
    st.subheader(selected)

    items_df = st.session_state.data['items']
    items_df = items_df[items_df['product_type_name'] == selected].iloc[-18:]

    # Visualize items according to category:
    cols = st.columns(2)
    for i, row in enumerate(items_df.itertuples(index=False)):
        with cols[i % 2]:
            st.image(row.image_url, use_container_width=True)
            name_col, price_col = st.columns([2, 1])
            with name_col:
                prod_name = ' '.join(row.prod_name.split(' ')[:3])
                st.markdown(f"**{prod_name}**")
            with price_col:
                st.markdown(f"**${row.price:.2f}**")
            detail_desc = row.detail_desc
            if len(detail_desc) < 60:
                detail_desc = f"{detail_desc} {detail_desc}"
            if len(detail_desc) > 85:
                detail_desc = f"{detail_desc[:85]}..."
            st.caption(detail_desc)

            button_key = f"button_{selected}_{i}"
            state_key = f"button_state_{selected}_{i}"
            if state_key not in st.session_state:
                st.session_state[state_key] = False

            def on_click(state_key: str) -> None:
                st.session_state[state_key] = True
                pos = int(state_key.split('_')[-1])
                order_df = st.session_state.data['orders']
                new_order = pd.DataFrame({
                    'order_id': [len(order_df)],
                    'user_id': [USER_ID],
                    'item_id': [int(items_df.iloc[pos]['item_id'])],
                    'date': [pd.Timestamp.now().round('s')],
                })
                st.session_state.data['orders'] = pd.concat(
                    [order_df, new_order],
                    axis=0,
                    ignore_index=True,
                )

            if not st.session_state[state_key]:
                st.button(
                    "🛒 Order",
                    key=button_key,
                    on_click=on_click,
                    args=(state_key, ),
                )
            else:
                st.button("✅ Ordered!", key=button_key, disabled=True)
