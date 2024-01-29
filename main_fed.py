import copy
import pickle
import numpy as np
import pandas as pd  # type:ignore
from logging import Logger
import torch
from typing import Optional, Union, cast
from models.Nets import MLP, CNNCifar, CNNCoba, CNNMnist

from utils.options import args_parser, get_logger
from utils.train_utils import (
    dynamic_model_selector_and_saver,
    get_data,
    get_model,
    save_metrics_graphs,
)
from models.Update import LocalUpdate
from models.test import test_img
import os
from pathlib import Path


def main() -> None:
    # parse args
    args = args_parser()
    filename: str = __file__.split(os.sep)[-1].split(".")[0]
    logger: Logger = get_logger(args=args, filename=filename)

    logger.log(level=logger.level, msg=f"Log level: {args.log_level.upper()}")

    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)

    logger.debug("%s dataset loaded", args.dataset.upper())

    base_dir: Path = Path(
        "save",
        args.dataset,
        f"{args.model}_iid{args.iid}_num{args.num_users}_C{args.frac}_le{args.local_ep}",
        f"shard{args.shard_per_user}",
    )

    run_num: int = int(args.results_save[-1])

    for file in base_dir.glob(pattern="*"):
        if args.results_save[:-1] in file.as_posix():
            run_num += 1
        else:
            break

    args.results_save = f"{args.results_save[:-1]}{run_num}"

    base_dir = Path(base_dir, f"seed{args.seed}_{args.results_save}")

    logger.info("Base save directory: %s", base_dir)

    if not Path(base_dir, "fed").exists():
        Path(base_dir, "fed").mkdir(exist_ok=True, parents=True)

    dict_save_path: Path = Path(base_dir, "dict_users.pkl")
    with open(dict_save_path, "wb") as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build model
    logger.debug("Building Model")
    net_glob = get_model(args)
    logger.debug("Model built\n%s", net_glob)

    logger.debug("Setting model to training mode")
    net_glob.train()

    # training
    results_save_path: Path = Path(base_dir, "fed", "results.csv")

    loss_train = []
    net_best: Optional[Union[CNNCifar, CNNMnist, CNNCoba, MLP]] = None
    best_acc = None

    lr: float = args.lr
    results: list = []
    final_results: Optional[pd.DataFrame] = None

    logger.info("Starting training loop (%i epochs)", args.epochs)
    for _iter in range(args.epochs):
        loss_locals = []
        w_glob: Optional[dict] = None
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        logger.info("Round %3d, lr: %.3f, %s", _iter, lr, idxs_users)

        for idx in idxs_users:
            logger.debug("User %i local training", idx)
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users_train[idx]
            )
            logger.debug("\tcreating net_local")
            net_local = copy.deepcopy(net_glob)
            logger.debug("\tnet_local created")

            logger.debug("\ttraining to get w_local and loss")
            w_local, loss = local.train(net=net_local.to(args.device))
            logger.debug("\ttraining completed")

            logger.debug("\tadding loss to loss_locals")
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                logger.debug("\tcreated w_glob (during User %i)", idx)
                w_glob = copy.deepcopy(w_local)
            else:
                logger.debug("\tadding w_local[k] to each key k in w_glob[k]")
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]

        logger.debug("Modifying lr")
        lr *= args.lr_decay

        # update global weights
        logger.debug("Updating global weights")
        w_glob = cast(dict, w_glob)
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)

        # copy weight to net_glob
        logger.debug("Copying weights")
        net_glob.load_state_dict(w_glob)

        # print loss
        logger.debug("Calculating Loss")
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (_iter + 1) % args.test_freq == 0:
            logger.debug("Evaluating net_glob")
            net_glob.eval()

            logger.debug("Calculating acc_test and loss_test")
            acc_test, loss_test, f1_test, precision_test, recall_test = test_img(
                net_glob, dataset_test, args
            )  # type:ignore
            logger.info(
                "\tAvg loss: %.4f, Test loss: %.6f, Accuracy: %.3f, F1: %.4f, Precision: %.4f, Recall: %.4f",
                loss_avg,
                loss_test,
                acc_test,
                f1_test,
                precision_test,
                recall_test,
            )

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test

            # if (iter + 1) > args.start_saving:
            #     model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(_iter + 1))
            #     torch.save(net_glob.state_dict(), model_save_path)

            results.append(
                np.array(
                    [
                        _iter,
                        loss_avg,
                        loss_test,
                        acc_test,
                        f1_test,
                        precision_test,
                        recall_test,
                        best_acc,
                    ]
                )
            )
            final_results = pd.DataFrame(
                np.array(results),
                columns=[
                    "epoch",
                    "loss_avg",
                    "loss_test",
                    "acc_test",
                    "f1_test",
                    "precision_test",
                    "recall_test",
                    "best_acc",
                ],
            )

            final_results.to_csv(results_save_path, index=False)

        if (_iter + 1) % 50 == 0:
            best_save_path: Path = Path(base_dir, "fed", f"best_{_iter+1}.pt")
            model_save_path: Path = Path(base_dir, "fed", f"model_{_iter+1}.pt")

            net_best = cast(Union[CNNCifar, CNNMnist, CNNCoba, MLP], net_best)
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)

    save_metrics_graphs(base_dir=base_dir, df=final_results)
    dynamic_model_selector_and_saver(base_dir=base_dir, args=args)


if __name__ == "__main__":
    main()
