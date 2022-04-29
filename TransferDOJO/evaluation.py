import torch
import ray
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import json
import glob
import os
import click
import pandas as pd


from UltraVision import data, models


def evaluate(model, test_loader, train_strategy):

    if train_strategy in ["train_classification"]:
        return evaluate_classification(model, test_loader)
    if train_strategy == "train_simclr":
        return evaluate_simclr(model, test_loader)
    if train_strategy == "train_classification_tune":
        return evaluate_classification_tune(model, test_loader)


def evaluate_classification(model, test_loader):
    model.eval()
    y_true = torch.tensor([], dtype=torch.long).cuda()
    pred_probs = torch.tensor([]).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        running_loss = 0.0
        for X, y in test_loader:
            inputs = X.cuda()
            labels = y.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            y_true = torch.cat((y_true, labels), 0)
            pred_probs = torch.cat((pred_probs, outputs), 0)

    # compute predicitions form probs
    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(pred_probs, 1)

    y_pred = y_pred.cpu().numpy()
    pred_probs = torch.nn.functional.softmax(pred_probs, dim=1).cpu().numpy()
    # get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    # macro auc score
    report["auc"] = roc_auc_score(
        y_true, pred_probs, multi_class="ovo", average="macro"
    )
    report["loss"] = running_loss / len(test_loader)
    print("Confusion Matrix: ")
    print(confusion_matrix(y_true, y_pred))

    return report


def evaluate_classification_tune(model, test_loader):
    # Validation loss
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    y_true = torch.tensor([], dtype=torch.long).cuda()
    pred_probs = torch.tensor([]).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    for i, (X, y) in enumerate(test_loader, 0):
        with torch.no_grad():
            inputs, labels = X, y
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            pred_probs = torch.cat((pred_probs, outputs), 0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1
            y_true = torch.cat((y_true, labels), 0)

    _, y_pred = torch.max(pred_probs, 1)
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    pred_probs = torch.nn.functional.softmax(pred_probs, dim=1).cpu().numpy()
    report = {}
    report["auc"] = roc_auc_score(
        y_true, pred_probs, multi_class="ovo", average="macro"
    )
    report["loss"] = val_loss / val_steps
    report["accuracy"] = correct / total
    print("Confusion Matrix: ")
    print(confusion_matrix(y_true, y_pred))
    return report


@click.command()
@click.option("--label_dir", default="", help="label location")
@click.option("--data_dir", default="", help="data folder")
@click.option("--data_name", default="", help="name of dataset")
@click.option("--val_size", default=0.1, type=float, help="size of validation set")
@click.option("--checkpoint_dir", default="", help="directory of checkpoints")
@click.option(
    "--checkpoint_num", default=14, type=int, help="checkpoint number to load"
)
@click.option("--model_name", default="", help="name of model")
@click.option("--save_results", default=None, help="location to save results")
@click.option(
    "--use_og_split", default=True, type=bool, help="use split given by dataset"
)
@click.option(
    "--normalize_option", default=None, type=str, help='either "paper", or "custom"'
)
@click.option(
    "--one_channel", default=True, type=bool, help="apply avg input layer operation"
)
def evaluate_tuned_models(
    label_dir,
    data_dir,
    data_name,
    val_size,
    checkpoint_dir,
    checkpoint_num,
    model_name,
    save_results,
    use_og_split,
    normalize_option,
    one_channel,
):
    # load data #
    train_loader, test_loader, val_loader, num_outputs = data.__dict__[data_name](
        label_dir,
        data_dir,
        val_size,
        batch_size=32,
        one_channel=one_channel,
        use_og_split=use_og_split,
        normalize_option=normalize_option,
    )
    # load model
    model = models.__dict__[model_name](
        pretrained=False, num_outputs=num_outputs, one_channel=one_channel
    )
    model = model.cuda()
    outcomes = {}
    outcomes_list = []
    training_seshes = glob.glob(os.path.join(checkpoint_dir, "*/"))
    for i, train in enumerate(training_seshes):
        try:
            checkpoint = os.path.join(
                train,
                f"checkpoint_{''.join(['0' for _ in range(6 - len(str(checkpoint_num)))] + [str(checkpoint_num)])}",
                "checkpoint",
            )
            model_state, _ = torch.load(checkpoint)
            model.load_state_dict(model_state)
        except:
            continue
        outcomes[i] = {}
        outcomes[i]["test"] = evaluate_classification(model, test_loader)
        outcomes_list.append(
            [
                i,
                "test",
                outcomes[i]["test"]["loss"],
                outcomes[i]["test"]["auc"],
                outcomes[i]["test"]["accuracy"],
            ]
        )
        outcomes[i]["train"] = evaluate_classification(model, train_loader)
        outcomes_list.append(
            [
                i,
                "train",
                outcomes[i]["test"]["loss"],
                outcomes[i]["test"]["auc"],
                outcomes[i]["test"]["accuracy"],
            ]
        )
        outcomes[i]["validation"] = evaluate_classification(model, val_loader)
        outcomes_list.append(
            [
                i,
                "validation",
                outcomes[i]["test"]["loss"],
                outcomes[i]["test"]["auc"],
                outcomes[i]["test"]["accuracy"],
            ]
        )

    if save_results:
        with open(
            os.path.join(save_results, f"eval_results_{str(normalize_option)}.json"),
            "w",
        ) as f:
            json.dump(outcomes, f)
        df = pd.DataFrame(
            outcomes_list,
            columns=["train_num", "data_split", "loss", "auc", "accuracy"],
        )
        df.to_csv(
            os.path.join(save_results, f"eval_results_{str(normalize_option)}.csv")
        )


@click.command()
@click.option("--label_dir", default="", help="label location")
@click.option("--data_dir", default="", help="data folder")
@click.option("--data_name", default="", help="name of dataset")
@click.option("--val_size", default=0.1, type=float, help="size of validation set")
@click.option("--checkpoint", default="", help="path to check point")
@click.option("--model_name", default="", help="name of model")
@click.option("--save_results", default=None, help="location to save results")
@click.option(
    "--use_og_split", default=True, type=bool, help="use split given by dataset"
)
@click.option(
    "--normalize_option", default=None, type=str, help='either "paper", or "custom"'
)
@click.option(
    "--one_channel", default=True, type=bool, help="apply avg input layer operation"
)
def evaluate_model(
    label_dir,
    data_dir,
    data_name,
    val_size,
    checkpoint,
    model_name,
    save_results,
    use_og_split,
    normalize_option,
    one_channel,
):
    # load data #
    train_loader, test_loader, val_loader, num_outputs = data.__dict__[data_name](
        label_dir,
        data_dir,
        val_size,
        batch_size=32,
        one_channel=one_channel,
        use_og_split=use_og_split,
        normalize_option=normalize_option,
    )
    # load model
    model = models.__dict__[model_name](
        pretrained=False, num_outputs=num_outputs, one_channel=one_channel
    )
    model_state = torch.load(checkpoint)
    model.load_state_dict(model_state)
    model = model.cuda()
    outcomes = {}
    outcomes_list = []

    outcomes["test"] = evaluate_classification_tune(model, test_loader)
    outcomes_list.append(
        [
            "test",
            outcomes["test"]["loss"],
            outcomes["test"]["auc"],
            outcomes["test"]["accuracy"],
        ]
    )
    outcomes["train"] = evaluate_classification_tune(model, train_loader)
    outcomes_list.append(
        [
            "train",
            outcomes["test"]["loss"],
            outcomes["test"]["auc"],
            outcomes["test"]["accuracy"],
        ]
    )
    outcomes["validation"] = evaluate_classification_tune(model, val_loader)
    outcomes_list.append(
        [
            "validation",
            outcomes["test"]["loss"],
            outcomes["test"]["auc"],
            outcomes["test"]["accuracy"],
        ]
    )

    if save_results:
        with open(
            os.path.join(save_results, f"eval_results_{str(normalize_option)}.json"),
            "w",
        ) as f:
            json.dump(outcomes, f)
        df = pd.DataFrame(
            outcomes_list, columns=["data_split", "loss", "auc", "accuracy"]
        )
        df.to_csv(
            os.path.join(save_results, f"eval_results_{str(normalize_option)}.csv")
        )
