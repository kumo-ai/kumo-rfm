import argparse

import pandas as pd
from kumoai.experimental import rfm
from kumoapi.rfm import InferenceConfig
from kumoapi.task import TaskType
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score

NUM_NEIGHBORS = {
    ("rel-hm", "user-churn"): [16, 16],
    ("rel-amazon", "user-churn"): [16, 16],
    ("rel-amazon", "item-churn"): [16, 16],
    ("rel-f1", "driver-top3"): [16, 16],
    ("rel-f1", "driver-dnf"): [16, 16],
    ("rel-event", "user-repeat"): [16, 16],
    ("rel-event", "user-ignore"): [16, 16],
    ("rel-stack", "user-engagement"): [16, 16],
    ("rel-stack", "user-badge"): [16, 16],
    ("rel-ratebeer", "beer-churn"): [10, 10],
    ("rel-ratebeer", "user-churn"): [16, 16],
    ("rel-ratebeer", "brewer-dormant"): [32],
    ('rel-mimic', 'patient-iculengthofstay'): [16, 16],
    ('rel-arxiv', 'paper-citation'): [16, 16],
}

CONTEXT_STRATEGY = {
    ("rel-hm", "user-churn"): "random",
    ("rel-amazon", "user-churn"): "random",
    ("rel-amazon", "item-churn"): "random",
    ("rel-f1", "driver-top3"): "random",
    ("rel-f1", "driver-dnf"): "random",
    ("rel-event", "user-repeat"): "recency",
    ("rel-event", "user-ignore"): "recency",
    ("rel-stack", "user-engagement"): "random",
    ("rel-stack", "user-badge"): "random",
    ("rel-ratebeer", "beer-churn"): "stratified_random",
    ("rel-ratebeer", "user-churn"): "random",
    ("rel-ratebeer", "brewer-dormant"): "random",
    ("rel-mimic", "patient-iculengthofstay"): "random",
    ("rel-arxiv", "paper-citation"): "stratified_random",
}

PQ = {
    ("rel-hm", "user-churn"):  #
    ("PREDICT COUNT(transactions.*, 0, 7, days)=0 "
     "FOR EACH customer.customer_id "
     "WHERE COUNT(transactions.*, -7, 0, days)>0"),
    ("rel-amazon", "user-churn"):  #
    ("PREDICT COUNT(review.*, 0, 91, days)=0 "
     "FOR EACH customer.customer_id "
     "WHERE COUNT(review.*, -91, 0, days)>0"),
    ("rel-amazon", "item-churn"):  #
    ("PREDICT COUNT(review.*, 0, 91, days)=0 "
     "FOR EACH product.product_id "
     "WHERE COUNT(review.*, -91, 0, days)>0"),
    ("rel-f1", "driver-top3"):  #
    ("PREDICT MIN(qualifying.position, 0, 30, days)<=3 "
     "FOR EACH drivers.driverId"),
    ("rel-f1", "driver-dnf"):  #
    ("PREDICT COUNT(results.* WHERE results.statusId != 1, 0, 30, days)>0 "
     "FOR EACH drivers.driverId "
     "ASSUMING COUNT(results.*, 0, 30, days)>0"),
    ("rel-event", "user-repeat"):  #
    ("PREDICT COUNT(event_attendees.* "
     "WHERE event_attendees.status IN ('yes', 'maybe'), 0, 7, days)>0 "
     "FOR EACH users.user_id "
     "WHERE COUNT(event_attendees.* "
     "WHERE event_attendees.status IN ('yes', 'maybe'), -14, 0, days)>0 "
     "ASSUMING COUNT(event_attendees.*, 0, 7, days) > 0"),
    ("rel-event", "user-ignore"):  #
    ("PREDICT COUNT(event_attendees.* "
     "WHERE event_attendees.status = 'invited', 0, 7, days)>2"
     "FOR EACH users.user_id "
     "ASSUMING COUNT(event_attendees.*, 0, 7, days)>0"),
    ("rel-stack", "user-engagement"):  #
    ("PREDICT COUNT(votes.*, 0, 91, days)>0 OR "
     "        COUNT(posts.*, 0, 91, days)>0 OR "
     "        COUNT(comments.*, 0, 91, days)>0 "
     "FOR EACH users.Id "
     "WHERE COUNT(votes.*, -INF, 0)>0 OR "
     "      COUNT(posts.*, -INF, 0)>0 OR "
     "      COUNT(comments.*, -INF, 0)>0"),
    ("rel-stack", "user-badge"):  #
    ("PREDICT COUNT(badges.*, 0, 91, days)>0 "
     "FOR EACH users.Id"),
    ("rel-ratebeer", "beer-churn"):  #
    ("PREDICT COUNT(beer_ratings.*, 0, 90, days)=0 "
     "FOR EACH beers.beer_id "
     "WHERE COUNT(beer_ratings.*, -90, 0, days)>0"),
    ("rel-ratebeer", "user-churn"):  #
    ("PREDICT COUNT(beer_ratings.*, 0, 90, days)=0 "
     "FOR EACH users.user_id "
     "WHERE COUNT(beer_ratings.*, -90, 0, days)>0"),
    ("rel-ratebeer", "brewer-dormant"):  #
    ("PREDICT COUNT(beers.*, 0, 365, days)=0 "
     "FOR EACH brewers.brewer_id "
     "WHERE COUNT(beers.*, -365, 0, days)>0"),
}

CONTEXT_SAMPLING_STRATEGIES = [
    "recency",
    "random",
    "stratified_recency",
    "stratified_random",
    "best",
]


def build_task(
    model: rfm.KumoRFM,
    dataset: str,
    task_name: str,
    context_sampling_strategy: str,
    context_size: int,
    lag_timesteps: int,
    batch_size: int,
    max_test_steps: int,
) -> rfm.TaskTable:
    task = get_task(dataset, task_name)
    task_type = TaskType(task.task_type.value)
    assert task_type == TaskType.BINARY_CLASSIFICATION

    columns = {
        task.entity_col: "ENTITY",
        task.time_col: "TIME",
        task.target_col: "TARGET",
    }

    context_df = pd.concat(
        [
            task.get_table("train", mask_input_cols=False).df,
            task.get_table("val", mask_input_cols=False).df,
        ],
        ignore_index=True,
    ).rename(columns=columns)

    if context_sampling_strategy == "best":
        context_sampling_strategy = CONTEXT_STRATEGY[(dataset, task_name)]
    if context_sampling_strategy == "recency":
        context_df = context_df.sort_values(
            ["TIME", "ENTITY"],
            ascending=False,
        ).iloc[:context_size].reset_index(drop=True)
    elif context_sampling_strategy == "random":
        context_df = context_df.sample(
            n=min(len(context_df), context_size),
            random_state=42,
        ).reset_index(drop=True)
    else:
        assert context_sampling_strategy.startswith("stratified")
        counts = context_df["TARGET"].value_counts()
        minority, majority = counts.idxmin(), counts.idxmax()
        if context_sampling_strategy == "stratified_recency":
            context_df = context_df.sort_values(
                ["TIME", "ENTITY"],
                ascending=False,
            )
        else:
            assert context_sampling_strategy == "stratified_random"
            context_df = context_df.sample(frac=1.0, random_state=42)
        minority_df = context_df[context_df["TARGET"] == minority]
        minority_df = minority_df.iloc[:context_size // 2]
        majority_df = context_df[context_df["TARGET"] == majority]
        majority_df = majority_df.iloc[:context_size - len(minority_df)]
        context_df = pd.concat([minority_df, majority_df], ignore_index=True)

    pred_df = task.get_table("test", mask_input_cols=False).df
    pred_df = pred_df.sample(
        n=min(len(pred_df), batch_size * max_test_steps),
        random_state=42,
    ).reset_index(drop=True).rename(columns=columns)

    task_table = rfm.TaskTable(
        task_type=task_type,
        context_df=context_df,
        pred_df=pred_df,
        entity_table_name=task.entity_table,
        entity_column="ENTITY",
        target_column="TARGET",
        time_column="TIME",
    )

    if (dataset, task_name) in PQ and lag_timesteps > 0:
        task_table = model.add_lagged_target(
            task_table,
            query=PQ[(dataset, task_name)],
            lag_timesteps=lag_timesteps,
        )
    elif lag_timesteps > 0:
        print(
            f"Lag labels not available for dataset={dataset} task={task_name}")

    return task_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--context_size", type=int, default=5_000)
    parser.add_argument(
        "--context_sampling_strategy",
        type=str,
        default="random",
        choices=CONTEXT_SAMPLING_STRATEGIES,
    )
    parser.add_argument("--lag_timesteps", type=int, default=10)
    parser.add_argument("--num_estimators", type=int, default=1)
    parser.add_argument("--column_shuffle", action="store_true")
    parser.add_argument("--category_shuffle", action="store_true")
    parser.add_argument("--class_shuffle", action="store_true")
    parser.add_argument("--hop_shuffle", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_test_steps", type=int, default=10)
    args = parser.parse_args()

    rfm.init()
    graph = rfm.Graph.from_relbench(args.dataset)
    model = rfm.KumoRFM(graph)

    task_table = build_task(
        model=model,
        dataset=args.dataset,
        task_name=args.task,
        context_sampling_strategy=args.context_sampling_strategy,
        context_size=args.context_size,
        lag_timesteps=args.lag_timesteps,
        batch_size=args.batch_size,
        max_test_steps=args.max_test_steps,
    )

    inference_config = InferenceConfig.from_task_type(task_table.task_type)
    inference_config.num_estimators = args.num_estimators
    inference_config.column_shuffle = args.column_shuffle
    inference_config.category_shuffle = args.category_shuffle
    inference_config.hop_shuffle = args.hop_shuffle
    inference_config.class_shuffle = args.class_shuffle

    with model.batch_mode():
        result_df = model.predict_task(
            task=task_table,
            run_mode="best",
            num_neighbors=NUM_NEIGHBORS.get((args.dataset, args.task),
                                            [16, 16]),
            inference_config=inference_config,
        )

    y_true = task_table._pred_df["TARGET"].to_numpy().astype(float)
    y_pred = result_df["True_PROB"].to_numpy()
    auroc = roc_auc_score(y_true, y_pred)
    print(f"AUROC: {100 * auroc:.2f}")
