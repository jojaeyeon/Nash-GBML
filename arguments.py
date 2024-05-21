import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Nash Model-Agnostic Meta-Learning (NashMAML)')

    parser.add_argument('--datasource', type=int, default=1, help='1: sinusoid, 2: miniImageNet')
    parser.add_argument('--model_type', type=int, default=1, help='1: MAML, 2: TR-MAML, 3: Reptile, 4: Meta-SGD, 5: CAVIA')
    parser.add_argument('--penalty_type', type=int, default=0, help='0: Default, 1: Penalty term1, 2: Penalty term2, 3: Penalty term3')

    parser.add_argument('--train', action="store_true", help='True to train, False to test')
    parser.add_argument('--num_iterations', type=int, default=70000, help='number of meta-iterations')
    parser.add_argument('--meta_batch_size', type=int, default=25, help='number of tasks sampled per meta-update')
    parser.add_argument('--update_batch_size', type=int, default=5, help='number of examples used fo inner gradient update (K for K-shot learning)')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='outer-loop learning rate')
    parser.add_argument('--lr_meta_decay', type=float, default=0.9, help='decay factor for meta learning rate')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='inner-loop learning rate')
    parser.add_argument('--step', type=int, default=1, help='the number of inner-loop')

    parser.add_argument('--seed', type=int, default=4, help='The seed value for experiment reproducibility')

    # Log
    parser.add_argument('--plot_log', type=int, default=10, help='plotting interval during training')
    parser.add_argument('--print_log', type=int, default=1000, help='printing interval during training')
    parser.add_argument('--save_log', type=int, default=10000, help='saving interval during training')

    parser.add_argument('--resume', action="store_true", help='True to resume')
    parser.add_argument('--save_iter', type=int, default=30000, help='initial iteration number')

    # Hyperparameter for Model
    # 2. MAML
    parser.add_argument('--p_lr', type=float, default=0.00001, help='task-probability learning rate for TR-MAML')
    # 3. Reptile
    parser.add_argument('--epsilon', type=float, default=1.0, help='meta-learning rate for reptile, lr = epsilon / inner lr')
    # 5. CAVIA
    parser.add_argument('--num_context_params', type=int, default=4, help='number of context parameters, 100 for MiniImageNet')
    parser.add_argument('--context_in', nargs='+', default=[False, False, True, False, False], help='per layer, indicate if context params are added')

    # Hyperparameter for Penalty
    parser.add_argument('--weight1', type=float, default=2.0, help='loss weight for penalty 1')
    parser.add_argument('--weight2', type=float, default=1.0, help='loss weight for penalty 2 and 3')
    parser.add_argument('--weight3', type=float, default=0.0000001, help='very small constant to prevent division by zero for penalty 2 and 3')

    # Hyperparameter for Datasource
    # 1. Sinusoid
    parser.add_argument('--n_input', type=int, default=1, help='dimension of input')
    parser.add_argument('--n_output', type=int, default=1, help='dimension of output')

    # 2. MiniImageNet
    parser.add_argument('--n_way', type=int, default=5, help='number of object classes to learn')
    parser.add_argument('--k_shot', type=int, default=1, help='number of examples per class to learn from')
    parser.add_argument('--k_query', type=int, default=15, help='number of examples to evaluate on (in outer loop)')
    parser.add_argument('--data_path', type=str, default='./data/miniimagenet/', help='folder which contains image data')

    # Hyperparameter for Network
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers for dataloader')

    # 1. FC
    parser.add_argument('--n_hidden', nargs="+", type=int, default=[40, 40], help='dimension of hidden layer')

    # 2. Conv
    parser.add_argument('--num_filters', type=int, default=32, help='number of filters per conv-layer')
    parser.add_argument('--num_film_hidden_layers', type=int, default=0, help='number of hidden layers used for FiLM')
    parser.add_argument('--imsize', type=int, default=84, help='downscale images to this size')
    parser.add_argument('--nn_init', type=str, default='kaiming', help='initialisation type (kaiming, xavier, None)')
    parser.add_argument('--no_max_pool', action='store_true', help='turn off max pooling in CNN')

    # Test Environment
    parser.add_argument('--n_samples', type=int, default=5000, help='the number of task samples for test')
    parser.add_argument('--test_samples', type=int, default=1000, help='the number of data of each task for test')

    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args