import os
from pathlib import Path
from typing import List
from torchvision import datasets, transforms  # type:ignore
from torch.utils.data import random_split
from models.Nets import MLP, CNNCoba, CNNMnist, CNNCifar
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


def graph_adjusted(col_name: str, filename: str, df: pd.DataFrame) -> None:
    adjusted_col = (
        df.groupby(np.arange(len(df)) // 10).mean()[col_name].values
    )  # averaged every 10 epochs
    epochs = np.arange(len(df) // 10) * 10
    sns.lineplot(x=epochs, y=adjusted_col).set(
        title=filename.split(os.sep)[-1].split(".")[0].title(),
        xlabel="Epochs",
        ylabel="Value",
    )
    plt.savefig(filename)
    plt.clf()


def save_metrics_graphs(base_dir: Path, df: pd.DataFrame) -> None:
    col_names: List[str] = [
        col_name for col_name in df.columns.values if col_name != "epoch"
    ]

    assert len(col_names) == 7

    for col_name in col_names:
        graph_adjusted(
            col_name=col_name,
            filename=Path(base_dir, "fed", f"{col_name}.png").as_posix(),
            df=df,
        )
