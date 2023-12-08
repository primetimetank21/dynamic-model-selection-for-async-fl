from typing import Dict
from torchvision import datasets, transforms
from torch.utils.data import random_split
from models.Nets import MLP, CNNMnist, CNNCifar
from utils.sampling import iid, noniid
from utils.coba_dataset import COBA


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
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args=args)
            dict_users_test, rand_set_all = noniid(
                dataset_test,
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
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args=args)
            dict_users_test, rand_set_all = noniid(
                dataset_test,
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
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args=args)
            dict_users_test, rand_set_all = noniid(
                dataset_test,
                args=args,
                rand_set_all=rand_set_all,
            )
    elif args.dataset == "coba":
        coba_dataset: COBA = COBA(root="data/coba", download=True)

        ## Create training and testing data
        train_size: int = int(
            0.8 * len(coba_dataset)
        )  # maybe TODO: make the percentage customizable (part of `args`)
        test_size: int = len(coba_dataset) - train_size
        dataset_train, dataset_test = random_split(
            dataset=coba_dataset, lengths=[train_size, test_size]
        )

        if args.iid:
            dict_users_train: Dict[int, set] = iid(
                dataset_train.dataset, args.num_users
            )
            dict_users_test: Dict[int, set] = iid(dataset_test.dataset, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train.dataset, args=args)
            dict_users_test, rand_set_all = noniid(
                dataset_test.dataset,
                args=args,
                rand_set_all=rand_set_all,
            )

    else:
        exit("Error: unrecognized dataset")

    return dataset_train, dataset_test, dict_users_train, dict_users_test


# pylint: disable=fixme
# TODO: Make model for COBA
def get_model(args):
    if args.model == "cnn" and args.dataset in ["cifar10", "cifar100"]:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == "cnn" and args.dataset in ["mnist"]:
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == "mlp" and args.dataset in ["mnist"]:
        net_glob = MLP(dim_in=784, dim_out=args.num_classes).to(args.device)
    else:
        exit("Error: unrecognized model")
    print(net_glob)

    return net_glob
