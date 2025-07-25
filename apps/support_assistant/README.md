# Customer Support Assistant 

This is a Streamlit demo app that showcases how to use KumoRFM for customer support use cases. 

## Features
- Churn prediction: Probability a user will stop purchasing in the next 90 days
- LTV: Realized and predicted customer lifetime value
- Personalized recommendations: Top-K product recommendation based on purchase histroy
- Item substitution: Alternatives for a given product based on popularity and category 

## Dataset 
The app uses Kumo’s public online shopping dataset, loaded from s3://kumo-sdk-public/rfm-datasets/online-shopping/
- users.parquet – customer profiles
- orders.parquet – order history
- items.parquet – item metadata

## Requirements
1. install dependacies: 
```
pip install streamlit pandas kumoai python-dotenv
```
2. Generate your own API key from https://kumorfm.ai/api-keys
3. Save your API key in .env:
```
KUMO_API_KEY=<your_kumo_api_key_here>
```
4. Spin up the app
```
streamlit run app.py
```
