import pandas as pd
import streamlit as st

def display_recommendations(pred_df: pd.DataFrame, items_df: pd.DataFrame) -> None:
    """
    Display product recommendations in Streamlit using predictions, in rows of 3 items.

    Parameters:
    - pred_df: pd.DataFrame with columns 'CLASS' (item_id) and 'SCORE' (probability)
    """
    # Page setup
    st.set_page_config(page_title="Fashion Shop", layout="wide")
    st.title("Fashion Shop 👗")
    st.subheader("Hi Matthias, welcome back!")

    # Prepare predictions DataFrame
    df = pred_df.rename(columns={"CLASS": "item_id", "SCORE": "probability"})
    merged = pd.merge(
        df,
        items_df[["item_id", "prod_name", "image_url"]],
        on="item_id",
        how="left"
    )

    # Display in rows of 3
    for i, row in enumerate(merged.itertuples(index=False)):
        if i % 3 == 0:
            cols = st.columns(3)
        col = cols[i % 3]
        with col:
            st.image(row.image_url, use_container_width=True)
            st.markdown(f"**{row.prod_name}**")
            st.metric(
                label="Purchase Probability",
                value=f"{row.probability * 100:.1f}%"
            )
