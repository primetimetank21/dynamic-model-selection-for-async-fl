import argparse
import logging
from typing import Optional
from pathlib import Path


def args_parser() -> argparse.Namespace():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument("--epochs", type=int, default=10, help="rounds of training")
    parser.add_argument(
        "--train_test_same",
        type=int,
        default=0,
        help="use same data to train and test on (COBA dataset), 0 for same",
    )
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument(
        "--shard_per_user", type=int, default=2, help="classes per user"
    )
    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=5, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=10, help="local batch size: B")
    parser.add_argument("--bs", type=int, default=128, help="test batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="user",
        help="train-test split type, user or sample",
    )
    parser.add_argument("--grad_norm", action="store_true", help="use_gradnorm_avging")
    parser.add_argument(
        "--local_ep_pretrain",
        type=int,
        default=0,
        help="the number of pretrain local ep",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="learning rate decay per round"
    )

    # model arguments
    parser.add_argument("--model", type=str, default="mlp", help="model name")
    parser.add_argument(
        "--kernel_num", type=int, default=9, help="number of each kind of kernel"
    )
    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default="3,4,5",
        help="comma-separated kernel size to use for convolution",
    )
    parser.add_argument(
        "--norm", type=str, default="batch_norm", help="batch_norm, layer_norm, or None"
    )
    parser.add_argument(
        "--num_filters", type=int, default=32, help="number of filters for conv nets"
    )
    parser.add_argument(
        "--max_pool",
        type=str,
        default="True",
        help="Whether use max pooling rather than strided convolutions",
    )
    parser.add_argument(
        "--num_layers_keep", type=int, default=1, help="number layers to keep"
    )

    # other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for reproducible 'randomness' (default: 0)",
    )
    parser.add_argument("--log_level", type=str, default="info", help="level of logger")
    parser.add_argument("--dataset", type=str, default="mnist", help="name of dataset")
    parser.add_argument("--iid", action="store_true", help="whether i.i.d or not")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument(
        "--num_channels", type=int, default=3, help="number of channels of imges"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument(
        "--stopping_rounds", type=int, default=10, help="rounds of early stopping"
    )
    parser.add_argument("--verbose", action="store_true", help="verbose print")
    parser.add_argument(
        "--print_freq",
        type=int,
        default=100,
        help="print loss frequency during training",
    )
    parser.add_argument(
        "--test_freq", type=int, default=1, help="how often to test on val set"
    )
    parser.add_argument(
        "--load_fed",
        type=str,
        default="",
        help="define pretrained federated model path",
    )
    parser.add_argument(
        "--results_save", type=str, default="/", help="define fed results save folder"
    )
    parser.add_argument(
        "--start_saving", type=int, default=0, help="when to start saving models"
    )

    args = parser.parse_args()
    return args


def get_logger(*, args: argparse.Namespace, filename: str) -> logging.Logger:
    # Create logs directory
    logs_dir = Path(Path.cwd(), "logs", args.dataset.lower())
    if not logs_dir.exists():
        logs_dir.mkdir(exist_ok=True, parents=True)

    # Create log formatter
    log_formatter = logging.Formatter(
        "[%(asctime)s] %(filename)s :: %(levelname)-8s :: %(message)s"
    )

    # Create handler to write logs to file
    file_path: Path = Path(logs_dir, f"{filename}.log")
    file_handler: logging.FileHandler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)

    # Create handler to output logs to console
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    # Get log level from command line arguments
    log_level: str = args.log_level
    numeric_level: Optional[int] = getattr(logging, log_level.upper(), None)

    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Create the logger and add the log level with the created handlers
    logger: logging.Logger = logging.getLogger(filename)

    logger.setLevel(level=numeric_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
