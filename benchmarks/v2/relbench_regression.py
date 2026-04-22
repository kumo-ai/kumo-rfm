import argparse

import pandas as pd
from kumoai.experimental import rfm
from kumoapi.rfm import InferenceConfig
from kumoapi.task import TaskType
from relbench.tasks import get_task
from sklearn.metrics import mean_absolute_error

NUM_NEIGHBORS = {
    ("rel-avito", "ad-ctr"): [32, 32],
    ("rel-event", "user-attendance"): [10, 10],
    ("rel-f1", "driver-position"): [8, 8],
    ("rel-hm", "item-sales"): [8, 8],
    ("rel-stack", "post-votes"): [32, 32],
    ("rel-trial", "study-adverse"): [10, 10],
    ("rel-trial", "site-success"): [32, 32],
    ("rel-amazon", "user-ltv"): [16, 16],
    ("rel-amazon", "item-ltv"): [16, 16],
    ("rel-ratebeer", "user-count"): [48, 48],
    ("rel-arxiv", "author-publication"): [16, 16],
}

CONTEXT_STRATEGY = {
    ("rel-avito", "ad-ctr"): "recency",
    ("rel-event", "user-attendance"): "recency",
    ("rel-f1", "driver-position"): "random",
    ("rel-hm", "item-sales"): "random",
    ("rel-stack", "post-votes"): "random",
    ("rel-trial", "study-adverse"): "recency",
    ("rel-trial", "site-success"): "recency",
    ("rel-amazon", "user-ltv"): "random",
    ("rel-amazon", "item-ltv"): "random",
    ("rel-ratebeer", "user-count"): "random",
    ("rel-arxiv", "author-publication"): "random",
}

PQ = {
    ("rel-avito", "ad-ctr"):  #
    ("PREDICT AVG(SearchStream.IsClick, 0, 4, days) "
     "FOR EACH AdsInfo.AdID "
     "ASSUMING SUM(SearchStream.IsClick, 0, 4, days)>0"),
    ("rel-event", "user-attendance"):  #
    ("PREDICT COUNT(event_attendees.* "
     "WHERE event_attendees.status IN ('yes', 'maybe'), 0, 7, days) "
     "FOR EACH users.user_id"),
    ("rel-f1", "driver-position"):  #
    ("PREDICT AVG(results.positionOrder, 0, 60, days) "
     "FOR EACH drivers.driverId "
     "WHERE COUNT(results.*, -365, 0, days)>0"),
    ("rel-hm", "item-sales"):  #
    ("PREDICT SUM(transactions.price, 0, 7, days) "
     "FOR EACH article.article_id"),
    ("rel-stack", "post-votes"):  #
    ("PREDICT COUNT(votes.*, 0, 91, days) "
     "FOR EACH posts.Id "),
    ("rel-trial", "study-adverse"):  #
    ("PREDICT SUM(reported_event_totals.subjects_affected "
     "WHERE reported_event_totals.event_type in ('serious', 'deaths'), "
     " 0, 365, days) "
     "FOR EACH studies.nct_id"),
    ("rel-amazon", "user-ltv"):  # 2 hops workaround with review price join
    ("PREDICT SUM(review.price, 0, 91, days) "
     "FOR EACH customer.customer_id "
     "WHERE COUNT(review.*, -91, 0, days)>0"),
    ("rel-amazon", "item-ltv"):  # 2 hops workaround with review price join
    ("PREDICT SUM(review.price, 0, 91, days) "
     "FOR EACH product.product_id"),
    ("rel-ratebeer", "user-count"):  #
    ("PREDICT COUNT(beer_ratings.*, 0, 90, days) "
     "FOR EACH users.user_id "
     "WHERE COUNT(beer_ratings.*, -90, 0, days)>0"),
    ("rel-arxiv", "author-publication"):  #
    ("PREDICT COUNT(paperAuthors.*, 0, 182, days) "
     "FOR EACH authors.Author_ID "),
}

CONTEXT_SAMPLING_STRATEGIES = [
    "recency",
    "random",
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
    assert task_type == TaskType.REGRESSION

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
    else:
        assert context_sampling_strategy == "random"
        context_df = context_df.sample(
            n=min(len(context_df), context_size),
            random_state=42,
        ).reset_index(drop=True)

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
    parser.add_argument("--hop_shuffle", action="store_true")
    parser.add_argument(
        "--target_transforms",
        type=str,
        nargs="+",
        default=["quantile"],
    )
    parser.add_argument("--output_type", type=str, default="median")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_test_steps", type=int, default=10)
    args = parser.parse_args()

    rfm.init()

    graph = rfm.Graph.from_relbench(args.dataset)
    if args.dataset == "rel-amazon":
        from kumoai.experimental.rfm.backend.local import LocalTable

        df_review = graph.tables["review"]._data
        df_product = graph.tables["product"]._data
        # this is necessary to make the pquery work
        df_joined = pd.merge(df_review,
                             df_product,
                             on="product_id",
                             how="left").drop(columns=["product_id"])
        df_review["price"] = df_joined["price"]
        graph.tables["review"] = LocalTable(
            df_review,
            name="review",
            time_column=graph.tables["review"].time_column.name,
        )

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
    inference_config.target_transforms = [
        t if t else None for t in args.target_transforms
    ]
    inference_config.output_type = args.output_type

    with model.batch_mode():
        result_df = model.predict_task(
            task=task_table,
            run_mode="best",
            num_neighbors=NUM_NEIGHBORS.get((args.dataset, args.task),
                                            [16, 16]),
            inference_config=inference_config,
        )

    y_true = task_table._pred_df["TARGET"].to_numpy().astype(float)
    y_pred = result_df["TARGET_PRED"].to_numpy()
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE: {mae:.4f}")
