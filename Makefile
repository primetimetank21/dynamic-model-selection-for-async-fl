install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black $$(git ls-files "*.py")

run_mnist_fedavg: install
	python main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save run1

run_cifar10_fedavg: install
	python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 50 --results_save run1

run_mnist_lgfedavg: install
	python main_lg.py --dataset mnist --model mlp --num_classes 10 --epochs 200 --lr 0.05 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --num_layers_keep 3 --results_save run1 --load_fed best_400.pt

run_cifar10_lgfedavg: install
	python main_lg.py --dataset cifar10 --model cnn --num_classes 10 --epochs 200 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 50 --num_layers_keep 2 --results_save run1 --load_fed best_1200.pt

run_mnist_mtl: install
	python main_mtl.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --num_layers_keep 5 --results_save run1

run_cifar10_mtl: install
	python main_mtl.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 50 --num_layers_keep 5 --results_save run1

run_all_mnist: run_mnist_fedavg run_mnist_lgfedavg run_mnist_mtl

run_all_cifar10: run_cifar10_fedavg run_cifar10_lgfedavg run_cifar10_mtl

run_all: run_all_mnist run_all_cifar10