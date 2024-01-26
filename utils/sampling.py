import random
from logging import Logger
import numpy as np
import torch
from typing import Any, Dict, Set, List, Optional, Union, cast
from argparse import Namespace
from utils.options import get_logger


def fair_iid(dataset, args: Namespace):
    """
    Sample I.I.D. client data from fairness dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_users = args.num_users
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(args.seed)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def fair_noniid(
    train_data,
    args: Namespace,
    num_shards=200,
    num_imgs=300,
    rand_set_all: Optional[List[set]] = None,
):
    """
    Sample non-I.I.D client data from fairness dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_users = args.num_users
    assert num_shards % num_users == 0
    shard_per_user = int(num_shards / num_users)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    # import pdb; pdb.set_trace()

    labels = (
        train_data[1]
        .numpy()
        .reshape(
            len(train_data[0]),
        )
    )
    assert num_shards * num_imgs == len(labels)
    # import pdb; pdb.set_trace()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    rand_set: Union[Set[Any], List[Set[Any]], None] = None

    # divide and assign
    if rand_set_all is None:
        rand_set_all = []
        for i in range(num_users):
            np.random.seed(args.seed)
            rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
            for rand in rand_set:
                rand_set_all.append(rand)

            idx_shard = list(
                set(idx_shard) - rand_set
            )  # remove shards from possible choices for other users
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

    else:  # this only works if the train and test set have the same distribution of labels
        for i in range(num_users):
            rand_set = rand_set_all[i * shard_per_user : (i + 1) * shard_per_user]
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

    return dict_users, rand_set_all


def iid(dataset, args: Namespace) -> Dict[int, set]:
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_users = args.num_users
    num_items = int(len(dataset) / num_users)
    dict_users: Dict[int, set] = {}
    all_idxs: List[int] = [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(args.seed)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, args: Namespace, rand_set_all: Optional[np.ndarray] = None):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_users = args.num_users
    shard_per_user = args.shard_per_user
    logger: Logger = get_logger(args=args, filename="noniid")

    dict_users: Dict[int, np.ndarray] = {
        i: np.array([], dtype="int64") for i in range(num_users)
    }

    idxs_dict: Dict[Union[int, float, Any], Any] = {}

    for i in range(len(dataset)):
        label: Union[int, float, Any] = None
        # Tweak for CIFAR10 dataset
        if isinstance(dataset.targets[i], int) and "cifar" in args.dataset:
            label = torch.tensor(dataset.targets[i]).item()
        elif args.dataset == "coba":
            label = dataset.targets[i].argmax().item()
        else:
            label = dataset.targets[i].item()

        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(list(idxs_dict.keys())))
    shard_per_class = int(shard_per_user * num_users / num_classes)

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if rand_set_all is None:
        try:
            rand_set_list: List[int] = list(range(num_classes)) * shard_per_class
            random.seed(args.seed)
            random.shuffle(rand_set_list)
            rand_set_all = np.array(rand_set_list).reshape((num_users, -1))
        except ValueError as ve:
            logger.warning("ValueError: %s. Attempting to reshape...", ve)
            for n in range(num_users, 0, -1):
                try:
                    rand_set_all = np.array(rand_set_list).reshape((n, -1))

                    args.num_users = n
                    num_users = n
                    break
                except ValueError:
                    continue

        rand_set_all = cast(np.ndarray, rand_set_all)

        logger.info("New rand_set_all.shape: %s", rand_set_all.shape)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            np.random.seed(args.seed)
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    dict_users = {key: val for key, val in dict_users.items() if len(val)}

    test = []
    for value in dict_users.values():
        x = np.unique(np.array(dataset.targets)[value])
        assert (len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(dataset)
    assert len(set(list(test))) == len(dataset)

    return dict_users, rand_set_all


def noniid_replace(
    dataset,
    args: Namespace,
    num_users,
    shard_per_user,
    rand_set_all: Optional[List[set]] = None,
):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}

    idxs_dict: Dict[Union[int, float, Any], Any] = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))

    rand_set_all = [] if rand_set_all is None else rand_set_all

    if len(rand_set_all) == 0:
        for i in range(num_users):
            np.random.seed(args.seed)
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label: set = rand_set_all[i]
        rand_set: list = []
        for label in rand_set_label:
            # pdb.set_trace()
            np.random.seed(args.seed)
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for _, value in dict_users.items():
        assert (
            len(np.unique(torch.tensor(dataset.targets).numpy()[value]))
            == shard_per_user
        )

    return dict_users, rand_set_all
