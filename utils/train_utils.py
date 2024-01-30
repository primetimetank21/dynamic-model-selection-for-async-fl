from argparse import Namespace
import os
import resource
from pathlib import Path
from typing import Dict, List, Optional, Union, cast
import torch
from torchvision import datasets, transforms  # type:ignore
from torch.utils.data import random_split
from models.Nets import MLP, CNNCoba, CNNMnist, CNNCifar
from models.test import test_img
from utils.coba_dataset import COBA, COBA_Split
from utils.sampling import iid, noniid
import seaborn as sns  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import pandas as pd  # type:ignore


trans_mnist = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
trans_cifar10_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar10_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar100_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)
trans_cifar100_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)


def get_data(args):
    if args.dataset == "mnist":
        dataset_train = datasets.MNIST(
            "data/mnist/", train=True, download=True, transform=trans_mnist
        )
        dataset_test = datasets.MNIST(
            "data/mnist/", train=False, download=True, transform=trans_mnist
        )
        # sample users
        if args.iid:
            dict_users_train = iid(dataset=dataset_train, args=args)
            dict_users_test = iid(dataset=dataset_test, args=args)
        else:
            dict_users_train, rand_set_all = noniid(dataset=dataset_train, args=args)
            dict_users_test, rand_set_all = noniid(
                dataset=dataset_test,
                args=args,
                rand_set_all=rand_set_all,
            )
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            "data/cifar10", train=True, download=True, transform=trans_cifar10_train
        )
        dataset_test = datasets.CIFAR10(
            "data/cifar10", train=False, download=True, transform=trans_cifar10_val
        )
        if args.iid:
            dict_users_train = iid(dataset=dataset_train, args=args)
            dict_users_test = iid(dataset=dataset_test, args=args)
        else:
            dict_users_train, rand_set_all = noniid(dataset=dataset_train, args=args)
            dict_users_test, rand_set_all = noniid(
                dataset=dataset_test,
                args=args,
                rand_set_all=rand_set_all,
            )
    elif args.dataset == "cifar100":
        dataset_train = datasets.CIFAR100(
            "data/cifar100", train=True, download=True, transform=trans_cifar100_train
        )
        dataset_test = datasets.CIFAR100(
            "data/cifar100", train=False, download=True, transform=trans_cifar100_val
        )
        if args.iid:
            dict_users_train = iid(dataset=dataset_train, args=args)
            dict_users_test = iid(dataset=dataset_test, args=args)
        else:
            dict_users_train, rand_set_all = noniid(dataset=dataset_train, args=args)
            dict_users_test, rand_set_all = noniid(
                dataset=dataset_test,
                args=args,
                rand_set_all=rand_set_all,
            )
    elif args.dataset == "coba":
        coba_dataset = COBA(root="data/coba", download=True)

        # Create training and testing data (Subsets)
        train_size = int(
            0.8 * len(coba_dataset)
        )  # maybe TODO: make the percentage customizable (part of `args`)
        test_size = len(coba_dataset) - train_size
        dataset_train, dataset_test = random_split(
            dataset=coba_dataset, lengths=[train_size, test_size]
        )

        if args.train_test_same != 0:
            # Convert Subset -> Dataset
            dataset_train = COBA_Split(dataset=dataset_train)
            dataset_test = COBA_Split(dataset=dataset_test)
        else:
            # Use same data for both training and testing
            dataset_train = dataset_train.dataset
            dataset_test = dataset_test.dataset

        if args.iid:
            dict_users_train = iid(dataset=dataset_train, args=args)
            dict_users_test = iid(dataset=dataset_test, args=args)
        else:
            dict_users_train, rand_set_all = noniid(dataset=dataset_train, args=args)
            dict_users_test, rand_set_all = noniid(
                dataset=dataset_test,
                args=args,
                rand_set_all=rand_set_all,
            )

    else:
        exit("Error: unrecognized dataset")

    return dataset_train, dataset_test, dict_users_train, dict_users_test


def get_model(args):
    if args.model == "cnn":
        if args.dataset in ["cifar10", "cifar100"]:
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.dataset in ["mnist"]:
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.dataset in ["coba"]:
            net_glob = CNNCoba(args=args).to(args.device)
    elif args.model == "mlp" and args.dataset in ["mnist"]:
        net_glob = MLP(dim_in=784, dim_out=args.num_classes).to(args.device)
    else:
        exit("Error: unrecognized model")
    print(net_glob)

    return net_glob


def graph_adjusted(base_dir: Path, col_name: str, df: pd.DataFrame) -> None:
    graphs_dir: Path = Path(base_dir, "fed", "metrics_graphs")
    graphs_dir.mkdir(exist_ok=True)

    graph_save_file: Path = Path(graphs_dir, f"{col_name.title()}.pdf")

    # averaged every 10 epochs
    adjusted_col = df.groupby(np.arange(len(df)) // 10).mean()[col_name].values
    epochs = np.arange(len(df) // 10) * 10
    sns.lineplot(x=epochs, y=adjusted_col).set(
        title=col_name.title(),
        xlabel="Epochs",
        ylabel="Value",
    )
    plt.savefig(graph_save_file)
    plt.clf()


def save_metrics_graphs(base_dir: Path, df: pd.DataFrame) -> None:
    col_names: List[str] = [
        col_name for col_name in df.columns.values if col_name != "epoch"
    ]

    assert len(col_names) == 7

    for col_name in col_names:
        graph_adjusted(
            base_dir=base_dir,
            col_name=col_name,
            df=df,
        )


class ChosenModel:
    def __init__(
        self, path: Path, performance_metrics: Dict[str, float], main_metric: str
    ) -> None:
        self.path: Path = path
        self.performance_metrics: Dict[str, float] = performance_metrics
        self.main_metric: str = main_metric
        self.main_metric_value: float = performance_metrics[main_metric]


def _update_metric_value(old_val: float, new_val: float, metric: str) -> bool:
    return new_val < old_val if metric == "loss" else new_val > old_val


def _reformat_model_for_dataframe(model: ChosenModel) -> Dict[str, Union[str, float]]:
    model_dict: Dict[str, Union[str, float]] = {
        "name": model.path.as_posix().split(os.sep)[-1],
        "main_metric": model.main_metric,
    }
    model_dict.update(
        **{metric: value for metric, value in model.performance_metrics.items()},
        path=model.path.as_posix(),
    )
    return model_dict


def _get_best_models(
    args: Namespace,
    test_dataset: Union[CNNCifar, CNNMnist, CNNCoba, MLP],
    metrics: List[str],
    model_paths: List[Path],
) -> Dict[str, Optional[ChosenModel]]:
    best_models: Dict[str, Optional[ChosenModel]] = {metric: None for metric in metrics}

    # Change open file limits to get best models
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    # Load into memory
    model = get_model(args)

    for model_path in model_paths:
        model.load_state_dict(
            torch.load(model_path)
        ) if args.device.type != "cpu" else model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )

        # Test model
        model.eval()
        acc_test, loss_test, f1_test, precision_test, recall_test = test_img(
            model, test_dataset, args
        )  # type:ignore

        results: Dict[str, float] = {
            metric: value
            for metric, value in zip(
                metrics, [acc_test, loss_test, f1_test, precision_test, recall_test]
            )
        }

        # Compare to the current best models for each performance metric, replacing if necessary
        for metric, metric_value in results.items():
            if best_models[metric] is None or _update_metric_value(
                old_val=best_models[metric].main_metric_value,  # type:ignore
                new_val=metric_value,
                metric=metric,
            ):
                best_models[metric] = ChosenModel(
                    path=model_path, performance_metrics=results, main_metric=metric
                )

    return best_models


def dynamic_model_selector_and_saver(args: Namespace, base_dir: Path) -> None:
    """
    Currently only tested on COBA dataset!
    """

    # Location to save the best models
    best_models_save_file: Path = Path(base_dir, "fed", "best_models.csv")

    REMOVE_BEST: bool = True  # This is to remove the presumed "best" model files
    SORTED: bool = False  # This is if we want to sort the models in epoch order

    # Store paths of the models
    models_dir: Path = Path(base_dir, "fed")

    model_paths: List[Path] = (
        [model_file for model_file in models_dir.glob("*.pt")]
        if not SORTED
        else sorted(
            [model_file for model_file in models_dir.glob("*.pt")],
            key=lambda s: int(
                s.as_posix().split(os.sep)[-1].split("_")[-1].replace(".pt", "")
            ),
        )
    )

    if REMOVE_BEST:
        model_paths = np.array(model_paths)[
            list("best_" not in model_path.as_posix() for model_path in model_paths)
        ].tolist()

    # Get testing data
    _, test_dataset, _, _ = get_data(args)

    # Get Best Models
    metrics: List[str] = ["accuracy", "loss", "f1", "precision", "recall"]
    best_models: Dict[str, Optional[ChosenModel]] = _get_best_models(
        args=args, test_dataset=test_dataset, metrics=metrics, model_paths=model_paths
    )

    # Save the Best Models
    models_dataframe_list: List[Dict[str, Union[str, float]]] = [
        _reformat_model_for_dataframe(model=cast(ChosenModel, model))
        for model in best_models.values()
    ]
    best_models_df = pd.DataFrame(models_dataframe_list)
    best_models_df.to_csv(path_or_buf=best_models_save_file, index=False, header=True)
