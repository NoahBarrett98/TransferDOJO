import click
import torch
import pandas as pd
import numpy as np
import json
import mlflow
import os
from tensorboardX import SummaryWriter
from datetime import datetime
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


from functools import partial
from sklearn.preprocessing import LabelEncoder

from TransferDOJO import data
from TransferDOJO import models
from TransferDOJO import train
from TransferDOJO import evaluation
from TransferDOJO import optimizers



def run_train_session(
    label_dir,
    data_dir,
    data_name,
    model_name,
    train_strategy,
    use_scheduler,
    batch_size,
    val_size,
    num_epochs,
    lr,
    optimizer_name,
    writer,
    one_channel=True,
    load_model_from=None,
    num_outputs_pretrained=2,
    split_seed=10,
    freeze_base=False,
    snapshot_dir=None,
    num_snapshots=1
):


    # preloaded data to prevent reloading
    train_loader, test_loader, val_loader, num_outputs = data.__dict__[data_name](
        label_dir,
        data_dir,
        val_size,
        batch_size,
        one_channel=one_channel,
        split_seed=split_seed,
        snapshot_dir=snapshot_dir
    )
    if snapshot_dir:
        # INIT ReprD
        data_size = len(train_loader)
        reprd = ReprD(
            data_size=data_size,
            num_epochs=num_epochs,
            num_snapshots=num_snapshots,
            save_path=snapshot_dir
        )
    else:
        reprd = None
    # load model
    model = models.__dict__[model_name](
        pretrained=True, num_outputs=num_outputs_pretrained, one_channel=one_channel
    )

    # get optimizer
    optimizer = optimizers.__dict__[optimizer_name](model, lr)
    # load dir #
    if load_model_from:
        model = models.load_base(model, load_model_from, num_outputs)
    if freeze_base:
        model = models.freeze_base(model)
    model = model.cuda()
    # set scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[200], gamma=0.5
        )
    else:
        scheduler = None


    # train model
    model = train.__dict__[train_strategy](
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        writer=writer,
        num_outputs=num_outputs,
        reprd=reprd
    )
    # evaluate model
    test_eval_results = evaluation.evaluate(model, test_loader, train_strategy)
    val_eval_results = evaluation.evaluate(model, val_loader, train_strategy)
    return test_eval_results, val_eval_results, model


def run_hparam_search(
    label_dir,
    data_dir,
    data_name,
    model_name,
    use_scheduler,
    batch_size,
    val_size,
    one_channel,
    num_epochs,
    optimizer_name,
    loaded_data,
    use_og_split,
    normalize_option,
    config,
    checkpoint_dir,
):
    # preloaded data to prevent reloading
    if loaded_data:
        (test_loader, train_loader, val_loader, num_outputs) = ray.get(loaded_data)
    else:
        # load data
        test_loader, train_loader, val_loader, num_outputs = data.__dict__[data_name](
            label_dir,
            data_dir,
            val_size,
            batch_size,
            one_channel=one_channel,
            use_og_split=use_og_split,
            normalize_option=normalize_option,
        )
    # load model
    model = models.__dict__[model_name](
        pretrained=True, num_outputs=num_outputs, one_channel=one_channel
    )
    model = model.cuda()
    # get optimizer
    optimizer = optimizers.__dict__[optimizer_name](model, config["lr"])
    # load dir #
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # set scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[200], gamma=0.5
        )
    else:
        scheduler = None

    # train model
    # CE for classification
    criterion = torch.nn.CrossEntropyLoss()
    # training
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # put on gpu
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            # outputs = torch.nn.functional.softmax(model(inputs), dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler:
            scheduler.step()

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
    # evaluate model
    # test_eval_results = evaluation.evaluate(model, test_loader, train_strategy="train_classification_tune")
    val_eval_results = evaluation.evaluate(
        model, val_loader, train_strategy="train_classification_tune"
    )

    tune.report(
        loss=val_eval_results["loss"],
        accuracy=val_eval_results["accuracy"],
        auc=val_eval_results["auc"],
    )


@click.command()
@click.option("--label_dir", default="", help="label location")
@click.option("--data_dir", default="", help="data folder")
@click.option("--data_name", default="", help="name of dataset")
@click.option("--model_name", default="", help="name of model")
@click.option(
    "--train_strategy", default="train_simclr", help="name of training strategy"
)
@click.option("--use_scheduler", default=True, help="whether to use scheduler")
@click.option("--batch_size", default=8, type=int, help="batch size for dataloaders")
@click.option("--num_epochs", default=100, type=int, help="number of epochs")
@click.option("--val_size", default=0.1, type=float, help="validation split size")
@click.option("--lr", default=0.01, type=float, help="optimizer learning rate")
@click.option("--optimizer_name", default="Adam", help="optimizer momentum")
@click.option("--save_results_dir", default=None, type=str, help="path to save model")
@click.option(
    "--one_channel",
    default=True,
    type=bool,
    help="wether to use 1 or 3 channels for input",
)
@click.option(
    "--load_model_from", default=None, type=str, help="load a pretrained model"
)
@click.option(
    "--num_outputs_pretrained",
    default=6,
    type=int,
    help="Number of outputs of pretrained model",
)
@click.option("--num_bootstraps", default=5, type=int, help="Number of bootstraps")
@click.option(
    "--freeze_base",
    default=False,
    type=bool,
    help="wether to freeze base for linear classification",
)
def train_bootstrap(
    label_dir,
    data_dir,
    data_name,
    model_name,
    train_strategy,
    use_scheduler,
    batch_size,
    val_size,
    num_epochs,
    lr,
    optimizer_name,
    save_results_dir,
    one_channel,
    load_model_from,
    num_outputs_pretrained,
    num_bootstraps,
    freeze_base,
):
    results = {}
    writer = None
    for seed in range(num_bootstraps):
        # train model
        print(f"TRAINING SEED {seed}:")
        test_eval_results, val_eval_results, model = run_train_session(
            label_dir,
            data_dir,
            data_name,
            model_name,
            train_strategy,
            use_scheduler,
            batch_size,
            val_size,
            num_epochs,
            lr,
            optimizer_name,
            writer,
            one_channel,
            load_model_from,
            num_outputs_pretrained,
            split_seed=seed,
            freeze_base=freeze_base,
        )
        results[seed] = {}
        results[seed]["test"] = test_eval_results
        results[seed]["val"] = val_eval_results

    if save_results_dir:
        with open(os.path.join(save_results_dir, "eval_results.json"), "w") as f:
            json.dump(results, f)


@click.command()
@click.option("--label_dir", default="", help="label location")
@click.option("--data_dir", default="", help="data folder")
@click.option("--data_name", default="", help="name of dataset")
@click.option("--model_name", default="", help="name of model")
@click.option(
    "--train_strategy", default="train_simclr", help="name of training strategy"
)
@click.option("--use_scheduler", default=True, help="whether to use scheduler")
@click.option("--batch_size", default=8, type=int, help="batch size for dataloaders")
@click.option("--num_epochs", default=100, type=int, help="number of epochs")
@click.option("--val_size", default=0.1, type=float, help="validation split size")
@click.option("--lr", default=0.01, type=float, help="optimizer learning rate")
@click.option("--momentum", default=0.9, type=float, help="optimizer weight_decay")
@click.option("--optimizer_name", default="Adam", help="optimizer momentum")
@click.option("--exp_name", default=None, help="name of experiment")
@click.option(
    "--tensorboard_name",
    default=None,
    type=str,
    help="wether to use tensorboard or not",
)
@click.option("--save_model_dir", default=None, type=str, help="path to save model")
@click.option(
    "--one_channel",
    default=True,
    type=bool,
    help="wether to use 1 or 3 channels for input",
)
@click.option(
    "--load_model_from", default=None, type=str, help="load a pretrained model"
)
@click.option(
    "--num_outputs_pretrained",
    default=2,
    type=int,
    help="Number of outputs of pretrained model",
)
@click.option(
    "--snapshot_dir",
    default=None,
    type=str,
    help="Directory to save snapshots for ReprDynamics",
)
@click.option(
    "--num_snapshots",
    default=1,
    type=int,
    help="number of snapshots for ReprDynamics",
)
def train_model(
    label_dir,
    data_dir,
    data_name,
    model_name,
    train_strategy,
    use_scheduler,
    batch_size,
    val_size,
    num_epochs,
    lr,
    momentum,
    optimizer_name,
    exp_name,
    tensorboard_name,
    save_model_dir,
    one_channel,
    load_model_from,
    num_outputs_pretrained,
    snapshot_dir,
    num_snapshots
):

    # setup tensorboard
    if tensorboard_name:
        if not os.path.exists(tensorboard_name):
            os.makedirs(tensorboard_name)
        tensorboard_location = os.path.join(
            tensorboard_name, f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H")}'
        )
        writer = SummaryWriter(tensorboard_location)
    else:
        tensorboard_location = "None"
        writer = None

    # setup mlflow experiment
    mlflow.set_experiment(exp_name)
    # train model
    test_eval_results, val_eval_results, model = run_train_session(
        label_dir,
        data_dir,
        data_name,
        model_name,
        train_strategy,
        use_scheduler,
        batch_size,
        val_size,
        num_epochs,
        lr,
        momentum,
        optimizer_name,
        writer,
        one_channel,
        load_model_from,
        num_outputs_pretrained,
        snapshot_dir,
        num_snapshots
    )
    # mlflow logging
    with mlflow.start_run():
        mlflow.log_param("data_name", data_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("train_strategy", train_strategy)
        mlflow.log_param("use_scheduler", use_scheduler)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("tboard loc", tensorboard_location)
        mlflow.log_metric("val_auc", val_eval_results["auc"])
        mlflow.log_metric("val_accuracy", val_eval_results["accuracy"])
        mlflow.log_metric("test_auc", test_eval_results["auc"])
        mlflow.log_metric("test_accuracy", test_eval_results["accuracy"])

    # save model
    if save_model_dir:
        models.save_model(model, save_model_dir)
        print(f"model saved to: {save_model_dir}")


@click.command()
@click.option("--label_dir", default="", help="label location")
@click.option("--data_dir", default="", help="data folder")
@click.option("--data_name", default="", help="name of dataset")
@click.option("--model_name", default="", help="name of model")
@click.option("--use_scheduler", default=True, help="whether to use scheduler")
@click.option("--batch_size", default=8, type=int, help="batch size for dataloaders")
@click.option("--num_epochs", default=100, type=int, help="number of epochs")
@click.option("--val_size", default=0.1, type=float, help="validation split size")
@click.option(
    "--one_channel", default=True, type=bool, help="apply avg input layer operation"
)
@click.option("--optimizer_name", default="SGD", help="optimizer momentum")
@click.option("--num_trials", default=5, type=int, help="number of trials")
@click.option(
    "--use_og_split", default=True, type=bool, help="use split given by dataset"
)
@click.option(
    "--normalize_option", default=None, type=str, help='either "paper", or "custom"'
)
@click.option(
    "--save_output_dir",
    default=None,
    type=str,
    help="save results of best trained model",
)
def hparam_search_wrapper(
    label_dir,
    data_dir,
    data_name,
    model_name,
    use_scheduler,
    batch_size,
    num_epochs,
    val_size,
    one_channel,
    optimizer_name,
    num_trials,
    use_og_split,
    normalize_option,
    save_output_dir,
):
    config = {"lr": tune.loguniform(1e-4, 1e-1)}

    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=num_epochs, grace_period=1, reduction_factor=2
    )
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "auc", "training_iteration"]
    )
    # run tuning experiment
    # load data once to prevent reloading
    loaded_data = ray.put(
        data.__dict__[data_name](
            label_dir, data_dir, val_size, batch_size, one_channel=one_channel
        )
    )

    def train_func(config, checkpoint_dir):
        run_hparam_search(
            label_dir=label_dir,
            data_dir=data_dir,
            data_name=data_name,
            model_name=model_name,
            use_scheduler=use_scheduler,
            batch_size=batch_size,
            val_size=val_size,
            one_channel=one_channel,
            num_epochs=num_epochs,
            optimizer_name=optimizer_name,
            loaded_data=loaded_data,
            use_og_split=use_og_split,
            normalize_option=normalize_option,
            config=config,
            checkpoint_dir=checkpoint_dir,
        )

    result = tune.run(
        train_func,
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
    )
    outcomes = {}
    outcomes["val"] = {}
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    outcomes["val"]["loss"] = best_trial.last_result["loss"]
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )
    outcomes["val"]["accuracy"] = best_trial.last_result["accuracy"]
    print("Best trial final validation auc: {}".format(best_trial.last_result["auc"]))
    outcomes["val"]["auc"] = best_trial.last_result["auc"]

    # load data
    train_loader, test_loader, val_loader, num_outputs = data.__dict__[data_name](
        label_dir,
        data_dir,
        val_size,
        batch_size,
        one_channel=one_channel,
        use_og_split=use_og_split,
        normalize_option=normalize_option,
    )
    # load model
    best_trained_model = models.__dict__[model_name](
        pretrained=True, num_outputs=num_outputs, one_channel=one_channel
    )
    # get best ckpt
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, "checkpoint")
    )

    best_trained_model.load_state_dict(model_state)
    best_trained_model = best_trained_model.cuda()

    outcomes["test"] = evaluation.evaluate(
        best_trained_model, test_loader, train_strategy="train_classification_tune"
    )
    print(
        f"Best trial test set accuracy: {outcomes['test']['accuracy']}, auc {outcomes['test']['auc']}, loss {outcomes['test']['loss']}"
    )
    if save_output_dir:
        with open(
            os.path.join(save_output_dir, f"best_results_{str(normalize_option)}.json"),
            "w",
        ) as f:
            json.dump(outcomes, f)
        torch.save(
            best_trained_model.state_dict(),
            os.path.join(save_output_dir, f"model_{str(normalize_option)}.pt"),
        )



