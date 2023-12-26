install:
	./.github/add_github_hooks.sh
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black $$(git ls-files "*.py")

lint:
	pylint --disable=R,C $$(git ls-files "*.py")

test:
	echo "TODO: implement tests":

jupyter:
	jupyter lab

run_coba_fedavg_default:
	python main_fed.py --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save coba_fedavg_default_run1

#testing purposes
run_coba_fedavg_ep1:
	python main_fed.py --dataset coba --model cnn --num_classes 14 --log_level debug --epochs 1 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save coba_fedavg_ep1_run1

#bad
run_coba_fedavg_decay:
	python main_fed.py --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.1 --lr_decay 0.95 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save coba_fedavg_decay_run1

#bad
run_coba_fedavg_decay_le10:
	python main_fed.py --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.1 --lr_decay 0.95 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 10 --local_bs 10 --results_save coba_fedavg_decay_le10_run1

#bad
run_coba_fedavg_decay_ep2000:
	python main_fed.py --dataset coba --model cnn --num_classes 14 --log_level info --epochs 2000 --lr 0.1 --lr_decay 0.95 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save coba_fedavg_decay_ep2000_run1

#bad
run_coba_fedavg_lower_lr:
	python main_fed.py --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.001 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save coba_fedavg_lower_lr_run1

#bad
run_coba_fedavg_lower_lr_and_decay:
	python main_fed.py --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.001 --lr_decay 0.95 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save coba_fedavg_lower_lr_and_decay_run1

#really good -- when loss func was Softmax (run1 under le10); trying with LogSoftmax (run2 and run3 under le10)
run_coba_fedavg_le10:
	python main_fed.py --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 10 --local_bs 10 --results_save coba_fedavg_le10_run1

run_mnist_fedavg:
	python main_fed.py --dataset mnist --model mlp --num_classes 10 --log_level info --epochs 1000 --lr 0.05 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save mnist_fedavg_run1

run_mnist_fedavg_le10:
	python main_fed.py --dataset mnist --model mlp --num_classes 10 --log_level info --epochs 1000 --lr 0.05 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 10 --local_bs 10 --results_save mnist_fedavg_le10_run1

run_cifar10_fedavg:
	python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --log_level info --epochs 2000 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 50 --results_save cifar10_fedavg_run1

run_cifar10_fedavg_le10:
	python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --log_level info --epochs 2000 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 10 --local_bs 50 --results_save cifar10_fedavg_le10_run1

run_mnist_lgfedavg:
	python main_lg.py --dataset mnist --model mlp --num_classes 10 --epochs 200 --lr 0.05 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --num_layers_keep 3 --results_save run1 --load_fed best_400.pt

run_cifar10_lgfedavg:
	python main_lg.py --dataset cifar10 --model cnn --num_classes 10 --epochs 200 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 50 --num_layers_keep 2 --results_save run1 --load_fed best_1200.pt

run_mnist_mtl:
	python main_mtl.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --num_layers_keep 5 --results_save run1

run_cifar10_mtl:
	python main_mtl.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 50 --num_layers_keep 5 --results_save run1

run_all_mnist: run_mnist_fedavg run_mnist_lgfedavg run_mnist_mtl

run_all_cifar10: run_cifar10_fedavg run_cifar10_lgfedavg run_cifar10_mtl

run_all: run_all_mnist run_all_cifar10

all: install format lint test
