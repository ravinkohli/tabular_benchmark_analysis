from __future__ import annotations
import pandas as pd
from dataclasses import dataclass


@dataclass
class Results:
    # Big ass predefined dictionary
    df: pd.DataFrame

    def at(
        self,
        *,
        method: str | list[str] | None = None,
        seed: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        metric: str | list[str] | None = None,
    ) -> Results:
        """Use this for slicing in to the dataframe to get what you need"""
        df = self.df
        items = {
            "method": method,
            "seed": seed,
        }
        for name, item in items.items():
            if item is None:
                continue
            idx: list = item if isinstance(item, list) else [item]
            df = df[df.index.get_level_values(name).isin(idx)]
            if not isinstance(item, list):
                df = df.droplevel(name, axis="index")

        if dataset:
            _dataset = dataset if isinstance(dataset, list) else [dataset]
            df = df.T.loc[df.T.index.get_level_values("dataset").isin(_dataset)].T
            if not isinstance(dataset, list):
                df = df.droplevel("dataset", axis="columns")

        if metric:
            _metric = metric if isinstance(metric, list) else [metric]
            df = df.T.loc[df.T.index.get_level_values("metric").isin(_metric)].T
            if not isinstance(metric, list):
                df = df.droplevel("metric", axis="columns")

        return Results(df)

    @property
    def methods(self) -> list[str]:
        return list(self.df.index.get_level_values("method").unique())

    @property
    def datasets(self) -> list[str]:
        return list(self.df.columns.get_level_values("dataset").unique())

    @property
    def metrics(self) -> list[str]:
        return list(self.df.columns.get_level_values("metric").unique())


def get_result(methods, metrics, dataset_names, result_df):
    headers = ["metric", "dataset"]
    indices = [
        "method",
        "seed",
    ]
    columns = pd.MultiIndex.from_product([metrics, dataset_names], names=headers)
    index = pd.MultiIndex.from_product([methods, [1]], names=indices)
    df = pd.DataFrame(columns=columns, index=index)
    df.sort_index(inplace=True)
    for index, row in result_df.iterrows():
        for method in methods:
            if int(row["dataset_id"]) not in dataset_names:
                continue
            if "logistic" in method:
                score_method = "linear"
            else:
                score_method = method
            row_id = (method, 1)
            col = ("acc", row["dataset_id"])
            df.loc[row_id, col] = row[f"score_{score_method}"]
    return Results(df=df)

def get_average_rank_table(results: Results):
    datasets = results.datasets
    metrics = sorted(results.metrics, reverse=True)
    # print(results.methods)
    df = results.df
    results_rank = {}
    results_score = {}
    for metric in metrics:
        if "time" in metric:
            continue
        metric_df = df[metric]
        dataset_rank_dfs = []
        dataset_mean_dfs = []
        for dataset in datasets:
            if dataset not in metric_df.columns:
                continue
            dataset_rank_df = metric_df[dataset].groupby('method').mean().rank(ascending=False)
            dataset_rank_dfs.append(dataset_rank_df)
            dataset_mean_dfs.append(metric_df[dataset])

        results_rank[metric.upper()] = pd.concat(dataset_rank_dfs).groupby("method").mean()
        
        results_score[metric.upper()] = pd.concat(dataset_mean_dfs).groupby("method").mean()
    score_df = pd.DataFrame(results_score).reset_index()
    rank_df = pd.DataFrame(results_rank).reset_index()
    final_table = rank_df.merge(score_df, on="method", suffixes=[" Mean Rank", " Mean Score"]).T
    final_table.columns = final_table.iloc[0]
    final_table = final_table.iloc[1:]
    return final_table

def pprint(df):
    for column in df:
        df[column] = df[column].astype('float').round(decimals=4)

    print(df.to_markdown())


def get_too_easy_select_acc_to_difference(df: pd.DataFrame, methods: list, stddev: float = 0.05):
    """ Selects the methods that are too easy on the criteria that the
    standard deviation of the scores is less than the given value
    """
    std_datasets = df[[f"score_{method}" for method in methods]].std(axis=1)
    selection_criteria = std_datasets < stddev
    too_easy_on_selection_criteria = df.loc[selection_criteria].index.to_list()
    select_on_selection_criteria = df.loc[list(map(lambda x: not x, selection_criteria))].index.to_list()
    return too_easy_on_selection_criteria, select_on_selection_criteria


def get_too_easy_select_acc_to_criteria(df: pd.DataFrame, better_methods: list, worse_methods: list):
    """
    Selects the methods that are too easy on the criteria that the worst of better methods is at
    least 5% better than the best of the worse methods"""
    lhs = df[better_methods].min(axis=1) if len(better_methods) > 1 else df[better_methods[0]]
    rhs = df[worse_methods].max(axis=1) if len(worse_methods) > 1 else df[worse_methods[0]]
    selection_criteria = lhs < 1.05 * rhs
    too_easy_on_selection_criteria = df.loc[selection_criteria].index.to_list()
    select_on_selection_criteria = df.loc[list(map(lambda x: not x, selection_criteria))].index.to_list()
    return too_easy_on_selection_criteria, select_on_selection_criteria