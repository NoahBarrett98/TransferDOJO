import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_theme()
"""
utils - mostly for visualization
"""


def parse_results_to_list(string, expname):
    results = []
    for s in string.split(" "):
        try:
            results.append(float(s))
        except ValueError:
            continue
    results = [results[pos : pos + 5] for pos in range(0, len(results), 5)]
    for i, r in enumerate(results):
        results[i] = r + [expname]
    return results


def final_results(json_f, out_dir):
    with open(json_f, "r") as f:
        final_results = json.load(f)
    total_results = []
    for k in final_results:
        total_results += parse_results_to_list(final_results[k], k)

    df = pd.DataFrame(
        total_results,
        columns=["lr", "loss", "accuracy", "auc", "training_iteration", "experiment"],
    )
    df = df.drop(columns=["training_iteration"])
    df.to_csv(out_dir)
    return df


def barplot(csv_dir, out_img):
    df = pd.read_csv(csv_dir)
    df = df[df["experiment"].isin(["custom_train", "custom_random"])]
    print(df.to_markdown())
    df = df.drop(columns=["lr", "loss"])
    df = pd.melt(df, id_vars=["experiment"], value_vars=["accuracy", "auc"])
    sns.boxplot(x="experiment", y="value", hue="variable", data=df)
    plt.suptitle("Result of Hyper-Parameter Search")
    plt.savefig(out_img)


def visualize_best_models(json_f, save_dir):
    with open(json_f, "r") as f:
        data = json.load(f)
    outcome = []
    for exp in data:
        for conf in data[exp]:
            for split in data[exp][conf]:
                for metric, value in data[exp][conf][split].items():
                    outcome.append([f"{exp}_{conf}", split, metric, value])
    data = pd.DataFrame(outcome, columns=["experiment", "split", "metric", "value"])
    data = data[data["experiment"].isin(["train_split_custom", "random_split_custom"])]
    import pdb

    pdb.set_trace()
    sns.barplot(
        x="metric", y="value", hue="experiment", data=data[data["split"] == "test"]
    )
    plt.suptitle("Best Test Results from Hyper Parameter Search")
    plt.savefig(os.path.join(save_dir, "test_barplot.png"))
    plt.clf()
    sns.barplot(
        x="metric", y="value", hue="experiment", data=data[data["split"] == "val"]
    )
    plt.suptitle("Best Val Results from Hyper Parameter Search")
    plt.savefig(os.path.join(save_dir, "val_barplot.png"))


def vis_bootstrap(json_files, save_dir):
    data = []
    for json_f in json_files:
        exp_name = json_f.split("/")[-2]
        with open(json_f, "r") as f:
            bootstrap = json.load(f)
        for seed in bootstrap:
            for split in bootstrap[seed]:
                data.append(
                    [
                        exp_name,
                        split,
                        bootstrap[seed][split]["accuracy"],
                        bootstrap[seed][split]["auc"],
                        bootstrap[seed][split]["loss"],
                    ]
                )
    df = pd.DataFrame(data, columns=["experiment", "split", "accuracy", "auc", "loss"])
    import pdb

    pdb.set_trace()
    df = df.drop(columns=["loss"])
    df = pd.melt(df, id_vars=["experiment", "split"], value_vars=["accuracy", "auc"])
    print(df)
    df.columns = ["experiment", "split", "metric", "value"]
    sns.boxplot(x="metric", y="value", hue="experiment", data=df[df["split"] == "test"])
    plt.suptitle("Linear Classification Proxy Method on Testing Set")
    plt.savefig(os.path.join(save_dir, "test_boxplot.png"))
    plt.clf()
    sns.boxplot(x="metric", y="value", hue="experiment", data=df[df["split"] == "val"])
    plt.suptitle("Linear Classification Proxy Method on Validation Set")
    plt.savefig(os.path.join(save_dir, "val_boxplot.png"))


