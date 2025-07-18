# KumoRFM: A foundation model for machine learning

## How you can use this repo
This repository is for tracking issues, feature requests, and feedback for KumoRFM.  
You'll also find a collection of sample Jupyter notebooks. You can download, study and modify in order to get hands-on with KumoRFM.

## Introduction
KumoRFM is built to simplify machine learning:
- Generate accurate predictions for tasks like churn, fraud, recommendations and many more
- Eliminate manual feature engineering and model training with zero-shot predictions
- Use just your data and a few lines of code to get results in real-time

## How KumoRFM works
KumoRFM is a pretrained foundation model designed for machine learning. It accepts a set of data tables as input and is prompted by a predictive task expressed in Predictive Query Language. The model delivers accurate predictions in under a second, without requiring task- or dataset-specific training. Benchmarks show it outperforms traditional supervised models by 2% to 8%.

KumoRFM is built on two core innovations: (1) A pretrained graph transformer: A table-agnostic encoder that learns representations across multi-modal, multi-table data, eliminating the need for custom pipelines. (2) In-context learning. At inference time, it retrieves labeled subgraph examples as context to inform predictions, eliminating the need for task-specific model training. 

[Blog](https://kumo.ai/company/news/kumo-relational-foundation-model/) | [Paper](https://kumo.ai/research/kumo_relational_foundation_model.pdf) | [Get an API key](https://kumorfm.ai/) 

<div align="center">
  <img src="https://kumo-sdk-public.s3.us-west-2.amazonaws.com/rfm-colabs/rfm-tasks.png"
       alt="Versatility of KumoRFM"
       width="600">
</div>


## Installation
KumoRFM is available for Python 3.9 to Python 3.13. You'll need an API key to use the model ([get an API key](https://kumorfm.ai/)).
```
pip install kumoai
```

## We love your feedback! :heart:
As you work with KumoRFM, if you encounter any problems or things that are confusing or don't work quite right, please open a new :octocat:[issue](https://github.com/kumo-ai/kumo-rfm/issues).

Join our [slack channel](https://join.slack.com/t/kumoaibuilders/shared_invite/zt-39jecw428-WYcsbIPJIpc80S2U5hSdyw)!

## Community contribution
If you're considering contributing a sample notebook, please first open a new :octocat:[issue](https://github.com/kumo-ai/kumo-rfm/issues) and state your proposed notebook so we discuss them together before you invest a ton of time. We'll invite you to our Mountain View, CA office (in you are local here) or send you a Kumo Swag if your notebook is accepted.

Thank you and excited to see what you'll build with KumoRFM!
