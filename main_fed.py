#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
from logging import Logger
import torch

from utils.options import args_parser, get_logger
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img
import os

# import pdb

if __name__ == "__main__":
    # pylint: disable=logging-fstring-interpolation
    # parse args
    args = args_parser()
    logger: Logger = get_logger(
        args=args, filename=__file__.split(os.sep)[-1].split(".")[0]
    )
    logger.log(level=logger.level, msg=f"Log level: {logger.level}")
    # logger.setLevel(level=logging.DEBUG if args.log_level == "debug" else logging.WARNING)
    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)

    if args.dataset == "coba":
        dataset_train, dataset_test = dataset_train.dataset, dataset_test.dataset

    logger.debug(f"{args.dataset.upper()} dataset loaded")

    base_dir = "./save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/".format(
        args.dataset,
        args.model,
        args.iid,
        args.num_users,
        args.frac,
        args.local_ep,
        args.shard_per_user,
        args.results_save,
    )
    if not os.path.exists(os.path.join(base_dir, "fed")):
        os.makedirs(os.path.join(base_dir, "fed"), exist_ok=True)

    dict_save_path = os.path.join(base_dir, "dict_users.pkl")
    with open(dict_save_path, "wb") as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build model
    logger.debug("Building Model")
    net_glob = get_model(args)
    logger.debug("Model built")

    logger.debug("Setting model to training mode")
    net_glob.train()

    # training
    results_save_path = os.path.join(base_dir, "fed/results.csv")

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []

    logger.debug("Starting training loop")
    for _iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, lr: {:.6f}, {}".format(_iter, lr, idxs_users))

        for idx in idxs_users:
            logger.debug(f"User {idx} local training")
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users_train[idx]
            )
            logger.debug("\tcreating net_local")
            net_local = copy.deepcopy(net_glob)
            logger.debug("\tnet_local created")

            logger.debug("\ttraining to get w_local and loss")
            w_local, loss = local.train(net=net_local.to(args.device))
            logger.debug("\ttraining completed")
            del net_local
            logger.debug("\tdeleted net_local")
            logger.debug("\tadding loss to loss_locals")
            loss_locals.append(copy.deepcopy(loss))
            logger.debug("\tloss added")
            del loss

            if w_glob is None:
                logger.debug(f"\tcreated w_glob for User {idx}")
                w_glob = copy.deepcopy(w_local)
            else:
                logger.debug("\tadding w_local[k] to each key k in w_glob[k]")
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]

        lr *= args.lr_decay

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (_iter + 1) % args.test_freq == 0:
            net_glob.eval()

            # pylint: disable=unbalanced-tuple-unpacking
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print(
                "Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}".format(
                    _iter, loss_avg, loss_test, acc_test
                )
            )

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = _iter

            # if (iter + 1) > args.start_saving:
            #     model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(_iter + 1))
            #     torch.save(net_glob.state_dict(), model_save_path)

            results.append(np.array([_iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(
                final_results,
                columns=["epoch", "loss_avg", "loss_test", "acc_test", "best_acc"],
            )
            final_results.to_csv(results_save_path, index=False)

        if (_iter + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, "fed/best_{}.pt".format(_iter + 1))
            model_save_path = os.path.join(
                base_dir, "fed/model_{}.pt".format(_iter + 1)
            )
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)

    print("Best model, iter: {}, acc: {}".format(best_epoch, best_acc))
