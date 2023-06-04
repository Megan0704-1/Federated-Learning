# split CIFAR10
cd drive/MyDrive/'Colab Notebooks'/'ML practice'/HW4/HFL/
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 1.0 --alpha 0.1 --n_user 10
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 1.0 --alpha 50.0 --n_user 10
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 1.0 --alpha 100.0 --n_user 10

# run (pwd: ./) 
# 1. Data distribution
# alpha50.0: 
python main.py --dataset CIFAR10-alpha50.0-ratio1.0-users10 --algorithm FedAvg --num_glob_iters 150 --local_epochs 10 --num_users 10 --learning_rate 0.1 --model resnet18 --device cuda --result_path alpha
# alpha0.1: 
python main.py --dataset CIFAR10-alpha0.1-ratio1.0-users10 --algorithm FedAvg --num_glob_iters 150 --local_epochs 10 --num_users 10 --learning_rate 0.1 --model resnet18 --device cuda --result_path alpha

python print_finish.py

# 2. Number of users in a round:
# 2 users: 
python main.py --dataset CIFAR10-alpha50.0-ratio1.0-users10 --algorithm FedAvg --num_glob_iters 150 --local_epochs 10 --num_users 2 --learning_rate 0.1 --model resnet18 --device cuda --result_path users
# 10 users: 
python main.py --dataset CIFAR10-alpha50.0-ratio1.0-users10 --algorithm FedAvg --num_glob_iters 150 --local_epochs 10 --num_users 10 --learning_rate 0.1 --model resnet18 --device cuda --result_path users
python print_finish.py


# 3. Model Accuracy
python main.py --dataset CIFAR10-alpha100.0-ratio1.0-users10 --algorithm FedAvg --num_glob_iters 150 --local_epochs 10 --num_users 10 --learning_rate 0.1 --model resnet18 --device cuda --result_path final
python print_finish.py
