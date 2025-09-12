# RelBench Run Guide

These two scripts provide an easy way to replicate the benchmarking results on RelBench.

## Run on a single dataset and task
```bash
python relbench_classification.py --dataset=rel-hm --task=user-churn
```

## Run all tasks on a single dataset
```bash
python relbench_classification.py --dataset=rel-amazon
```

## Run all tasks on all datasets
```bash
python relbench_classification.py
```
