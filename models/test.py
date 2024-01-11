import copy
from typing import Optional, Tuple, Union
import numpy as np
from scipy import stats
from utils.options import get_logger
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def test_img(
    net_g, datatest, args, return_probs: bool = False, user_idx: int = -1
) -> Union[
    Tuple[float, float, float, float, float],
    Tuple[float, float, float, float, float, torch.Tensor],
]:
    logger = get_logger(args=args, filename="test_img")

    net_g.eval()

    # testing
    logger.debug("Starting testing in test_img()")
    test_loss: float = 0
    accuracy: float = 0
    f1: float = 0
    precision: float = 0
    recall: float = 0
    class_labels: Optional[list] = (
        list(datatest.class_to_idx.keys()) if args.dataset == "coba" else None
    )

    logger.debug("Creating DataLoader")
    data_loader: DataLoader = DataLoader(datatest, batch_size=args.bs)

    probs: np.array = np.array([])
    IS_USING_GPU: bool = args.gpu != -1 and args.device.type != "cpu"

    logger.debug("Starting test loop: (len = %i)", len(data_loader))
    for _, (data, target) in enumerate(data_loader):
        if args.gpu != -1 and args.device.type != "cpu":
            data, target = data.to(args.device), target.to(args.device)
        if args.dataset == "coba":
            data = data.permute(0, 3, 1, 2)

        logger.debug("\tcalculating log_probs")
        log_probs: torch.Tensor = net_g(data)

        probs: np.array = (
            np.append(probs, log_probs.cpu().data.numpy())
            if IS_USING_GPU
            else np.append(probs, log_probs.data.numpy())
        )

        # Sum up batch loss
        logger.debug("\tcalculating cross entropy loss")
        t: torch.Tensor = target.to(torch.float32) if args.dataset == "coba" else target
        test_loss += F.cross_entropy(log_probs, t, reduction="sum").item()

        # Get predictions (y_pred)
        logger.debug("\tgetting predictions (y_preds)")
        y_pred: torch.Tensor = (
            log_probs.cpu().data.max(1, keepdim=True)[1]
            if args.device.type != "cpu"
            else log_probs.data.max(1, keepdim=True)[1]
        )

        # Get true labels (y_true)
        logger.debug("\tgetting labels (y_true)")
        y_true: torch.Tensor = (
            torch.tensor(
                list(map(torch.argmax, target.data)), device="cpu"
            ).data.view_as(y_pred)
            if args.dataset == "coba"
            else target.to("cpu").data.view_as(y_pred)
        )

        # Calculate Performance metrics
        logger.debug("\tcalculating performance metrics")
        accuracy += accuracy_score(y_pred=y_pred, y_true=y_true, normalize=False)
        f1 += f1_score(
            labels=class_labels,
            y_pred=y_pred,
            y_true=y_true,
            average="weighted",
            zero_division=0.0,
        )
        recall += recall_score(
            labels=class_labels,
            y_pred=y_pred,
            y_true=y_true,
            average="weighted",
            zero_division=0.0,
        )
        precision += precision_score(
            labels=class_labels,
            y_pred=y_pred,
            y_true=y_true,
            average="weighted",
            zero_division=0.0,
        )

    N: int = len(data_loader.dataset)

    logger.debug("Calculating accuracy")
    correct: int = accuracy
    accuracy /= N
    accuracy *= 100.00
    logger.debug("\taccuracy = %f", accuracy)

    logger.debug("Calculating test_loss")
    test_loss /= N
    logger.debug("\ttest_loss = %f", test_loss)

    logger.debug("Calculating f1-score")
    f1 /= N
    f1 *= 100.00
    logger.debug("\tf1 = %f", f1)

    logger.debug("Calculating precision")
    precision /= N
    precision *= 100.00
    logger.debug("\tprecision = %f", precision)

    logger.debug("Calculating recall")
    recall /= N
    recall *= 100.00
    logger.debug("\trecall = %f", recall)

    if args.verbose:
        if user_idx < 0:
            logger.info(
                "Test set: Avg loss: %.4f, Accuracy: %i/%i (%.2f%%), F1: %.4f, Precision: %.4f, Recall: %.4f",
                test_loss,
                correct,
                N,
                accuracy,
                f1,
                precision,
                recall,
            )
        else:
            logger.info(
                "Local model %i: Average loss: %.4f, Accuracy: %i/%i (%.2f%%), F1: %.4f, Precision: %.4f, Recall: %.4f",
                user_idx,
                test_loss,
                correct,
                N,
                accuracy,
                f1,
                precision,
                recall,
            )

    # pylint: disable=unbalanced-tuple-unpacking
    if return_probs:
        return accuracy, test_loss, f1, precision, recall, torch.cat(probs)
    return accuracy, test_loss, f1, precision, recall


def test_img_local(net_g, dataset, args, user_idx=-1, idxs=None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(
        DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=False
    )
    # l = len(data_loader)

    for _, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)

        if args.dataset == "coba":
            data = data.permute(0, 3, 1, 2)

        log_probs = net_g(data)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction="sum").item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if args.verbose:
        print(
            "Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                user_idx, test_loss, correct, len(data_loader.dataset), accuracy
            )
        )

    return accuracy, test_loss


def test_img_local_all(
    net_local_list, args, dataset_test, dict_users_test, return_all=False
):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        a, b = test_img_local(
            net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx]
        )

        acc_test_local[idx] = a
        loss_test_local[idx] = b

    if return_all:
        return acc_test_local, loss_test_local
    return acc_test_local.mean(), loss_test_local.mean()


def test_img_avg_all(net_glob, net_local_list, args, dataset_test, return_net=False):
    net_glob_temp = copy.deepcopy(net_glob)
    w_keys_epoch = net_glob.state_dict().keys()
    w_glob_temp = {}
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        w_local = net_local.state_dict()

        if len(w_glob_temp) == 0:
            w_glob_temp = copy.deepcopy(w_local)
        else:
            for k in w_keys_epoch:
                w_glob_temp[k] += w_local[k]

    for k in w_keys_epoch:
        w_glob_temp[k] = torch.div(w_glob_temp[k], args.num_users)
    net_glob_temp.load_state_dict(w_glob_temp)

    # pylint: disable=unbalanced-tuple-unpacking
    acc_test_avg, loss_test_avg = test_img(net_glob_temp, dataset_test, args)

    if return_net:
        return acc_test_avg, loss_test_avg, net_glob_temp
    return acc_test_avg, loss_test_avg


def test_img_ensemble_all(net_local_list, args, dataset_test):
    probs_all = []
    preds_all = []
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        # _, _, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        _, _, _, _, _, probs = test_img(
            net_local, dataset_test, args, return_probs=True, user_idx=idx
        )
        # print('Local model: {}, loss: {}, acc: {}'.format(idx, loss, acc))
        probs_all.append(probs.detach())

        preds = probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        preds_all.append(preds)

    labels = np.array(dataset_test.targets)
    preds_probs = torch.mean(torch.stack(probs_all), dim=0)

    # ensemble (avg) metrics
    criterion = nn.CrossEntropyLoss()
    preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
    loss_test = criterion(preds_probs, torch.tensor(labels).to(args.device)).item()
    acc_test_avg = (preds_avg == labels).mean() * 100

    # ensemble (maj)
    preds_all = np.array(preds_all).T
    preds_maj = stats.mode(preds_all, axis=1)[0].reshape(-1)
    acc_test_maj = (preds_maj == labels).mean() * 100

    return acc_test_avg, loss_test, acc_test_maj
