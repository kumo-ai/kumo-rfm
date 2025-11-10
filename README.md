<div align="center">
  <img src="https://kumo-ai.github.io/kumo-sdk/docs/_static/kumo-logo.svg" height="40"/>
  <h1>KumoRFM: A foundation model for business data</h1>
</div>

<div align="center">
  <p>
    <a href="https://kumo.ai/company/news/kumo-relational-foundation-model/">Blog</a> •
    <a href="https://kumo.ai/research/kumo_relational_foundation_model.pdf">Paper</a> •
    <a href="https://github.com/kumo-ai/kumo-rfm-mcp">MCP Server</a> •
    <a href="https://kumorfm.ai">Get an API key</a>
  </p>

  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kumoai?color=FC1373)](https://pypi.org/project/kumoai/)
  [![PyPI Status](https://img.shields.io/pypi/v/kumoai.svg?color=FC1373)](https://pypi.org/project/kumoai/)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/kumoai?color=FC1373)](https://pepy.tech/project/kumoai)
  [![Slack](https://img.shields.io/badge/discord-join-pink.svg?logo=discord&color=FC1373)](https://discord.gg/uNB4bJkapQ)

  This repository is for tracking issues, feature requests, feedbacks and examples for KumoRFM.
</div>

## Introduction
KumoRFM generates predictions from business data:
- Predict outcomes such as purchases, engagement, churn, fraud, revenue, and more.
- No predictive modeling work, no feature engineering, no waiting.
- Give tabular data as context, prompt with a SQL-like query, get predicted values in return.

## Quick start
Install Python SDK and get an API key to use KumoRFM at [kumorfm.ai](https://kumorfm.ai).
```
pip install kumoai
```
Quick start notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kumo-ai/kumo-rfm/blob/master/notebooks/quickstart.ipynb)

## 🔬 MCP Server

We provide a **KumoRFM MCP server** for integration with your agentic workflows.

See :octocat: https://github.com/kumo-ai/kumo-rfm-mcp for more information.

## How KumoRFM works
KumoRFM is trained to understand tabular data and predict new values and connections. You prompt it with a SQL-like query and give it tabular business data as context. It’ll understand the full meaning and relationships within your data, generate the necessary predictions to answer your query, and give you the answer in about a second. The predictions are as accurate or slightly more accurate than custom-trained predictive models that often take weeks to make.

KumoRFM is built on two core innovations: (1) A pretrained graph transformer: An encoder that learns representations across multiple tables, eliminating the need for custom pipelines. (2) In-context learning. At inference time, it retrieves labeled subgraph examples as context to inform predictions, eliminating the need for task-specific model training. 

<div align="center">
  <picture class="github-only">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/7539d4b7-144d-403b-b0c5-e9cb47c802c4">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/a8ec31a6-604b-461f-800c-4581045d49f8">
    <img src="https://github.com/user-attachments/assets/7539d4b7-144d-403b-b0c5-e9cb47c802c4" width="700">
  </picture>
</div>

## We love your feedback! :heart:
As you work with KumoRFM, if you encounter any problems or things that are confusing or don't work quite right, please open a new :octocat: [issue](https://github.com/kumo-ai/kumo-rfm/issues/new/choose). You can also submit general feedback and suggestions [here](https://docs.google.com/forms/d/e/1FAIpQLSfr2HYgJN8ghaKyvU0PSRkqrGd_BijL3oyQTnTxLrf8AEk-EA/viewform). Join [our Discord](https://discord.gg/uNB4bJkapQ)!

## Community contribution 🤝
If you're considering contributing an example notebook, please first open a new :octocat:[issue](https://github.com/kumo-ai/kumo-rfm/issues/new/choose) and state your proposed notebook so we discuss them together before you invest a ton of time. We'll invite you to our Mountain View, CA office (if you are local here) or send you a Kumo Swag if your notebook is accepted.

Thank you and excited to see what you'll build with KumoRFM!
