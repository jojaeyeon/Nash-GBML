import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import copy


def simplex_proj(beta):
    beta_sorted = np.flip(np.sort(beta))
    rho = 1
    for i in range(len(beta) - 1):
        j = len(beta) - i
        test = beta_sorted[j - 1] + (1 - np.sum(beta_sorted[:j])) / (j)
        if test > 0:
            rho = j
            break

    lam = (1 - np.sum(beta_sorted[:rho])) / (rho)
    return np.maximum(beta + lam, 0)


class MAMLModel(nn.Module):
    def __init__(self, n_input=1, n_output=1, n_hidden=[40, 40]):
        super(MAMLModel, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.layer = nn.ModuleList()

        self.layer.append(nn.Linear(n_input, n_hidden[0], bias=True))
        self.layer.append(nn.ReLU())
        for i in range(len(n_hidden) - 1):
            self.layer.append(nn.Linear(n_hidden[i], n_hidden[i + 1], bias=True))
            self.layer.append(nn.ReLU())
        self.layer.append(nn.Linear(n_hidden[-1], n_output, bias=True))

        self.reset_params()
        self.directory = "Sinusoid"

    def reset_params(self):
        for i in range(0, len(self.layer) + 1, 2):
            nn.init.trunc_normal_(self.layer[i].weight, mean=0, std=0.01, a=-0.02, b=0.02)
            nn.init.zeros_(self.layer[i].bias)

    def forward(self, x):
        for i in range(0, len(self.layer) - 1, 2):
            x = self.layer[i](x)
            x = F.relu(x)
        x = self.layer[-1](x)

        return x

    def parameterised(self, x, weights):
        for i in range(0, len(weights) - 2, 2):
            x = F.linear(x, weights[i], weights[i + 1])
            x = F.relu(x)
        x = F.linear(x, weights[-2], weights[-1])

        return x


class MAMLConvModel(nn.Module):
    def __init__(self, num_classes, num_filters, max_pool, imsize, initialisation="kaiming", device="cpu"):
        super(MAMLConvModel, self).__init__()

        self.num_classes = num_classes
        self.num_filters = num_filters
        self.max_pool = max_pool
        self.kernel_size = 3

        stride = 1
        padding = 1
        self.num_channels = 3

        self.conv1 = nn.Conv2d(self.num_channels, self.num_filters, self.kernel_size, stride=stride, padding=padding).to(device)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride, padding=padding).to(device)
        self.conv3 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride, padding=padding).to(device)
        if not self.max_pool:
            self.conv4 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride).to(device)
        else:
            self.conv4 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride, padding=padding).to(device)

        # batch norm
        self.bn1 = nn.BatchNorm2d(self.num_filters, track_running_stats=True).to(device)
        self.bn2 = nn.BatchNorm2d(self.num_filters, track_running_stats=True).to(device)
        self.bn3 = nn.BatchNorm2d(self.num_filters, track_running_stats=True).to(device)
        self.bn4 = nn.BatchNorm2d(self.num_filters, track_running_stats=True).to(device)

        # initialise weights for the fully connected layer
        if imsize == 84:
            self.fc1 = nn.Linear(5 * 5 * self.num_filters, self.num_classes).to(device)
        elif imsize == 28:
            self.fc1 = nn.Linear(self.num_filters, self.num_classes).to(device)
        else:
            raise NotImplementedError('Cannot handle image size.')

        # parameter initialisation (if different than standard pytorch one)
        if initialisation != 'standard':
            self.init_params(initialisation)

        self.directory = "MiniImageNet"

    def init_params(self, initialisation):
        # convolutional weights
        if initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu', self.conv1.weight))
            torch.nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu', self.conv2.weight))
            torch.nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu', self.conv3.weight))
            torch.nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('relu', self.conv4.weight))
        elif initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')

        # convolutional bias
        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)
        self.conv3.bias.data.fill_(0)
        self.conv4.bias.data.fill_(0)

        # fully connected weights at the end
        if initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('linear', self.fc1.weight))
        elif initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='linear')

        # fully connected bias
        self.fc1.bias.data.fill_(0)

    def forward(self, x):
        # pass through convolutional layer
        h1 = self.conv1(x)
        # batchnorm
        h1 = self.bn1(h1)
        # do max-pooling (for imagenet)
        if self.max_pool:
            h1 = F.max_pool2d(h1, kernel_size=2)
        # pass through ReLu activation function
        h1 = F.relu(h1)

        h2 = self.conv2(h1)
        h2 = self.bn2(h2)
        if self.max_pool:
            h2 = F.max_pool2d(h2, kernel_size=2)
        h2 = F.relu(h2)

        h3 = self.conv3(h2)
        h3 = self.bn3(h3)
        if self.max_pool:
            h3 = F.max_pool2d(h3, kernel_size=2)
        h3 = F.relu(h3)

        h4 = self.conv4(h3)
        h4 = self.bn4(h4)
        if self.max_pool:
            h4 = F.max_pool2d(h4, kernel_size=2)
        h4 = F.relu(h4)

        # flatten
        h4 = h4.view(h4.size(0), -1)

        y = self.fc1(h4)

        return y

    def parameterised(self, x, weights):
        h1 = F.conv2d(x, weight=weights[0], bias=weights[1], stride=1, padding=1)
        h1 = F.batch_norm(h1, running_mean=self.bn1.running_mean, running_var=self.bn1.running_var, weight=weights[8], bias=weights[9])
        if self.max_pool:
            h1 = F.max_pool2d(h1, kernel_size=2)
        h1 = F.relu(h1)

        h2 = F.conv2d(h1, weight=weights[2], bias=weights[3], stride=1, padding=1)
        h2 = F.batch_norm(h2, running_mean=self.bn2.running_mean, running_var=self.bn2.running_var, weight=weights[10], bias=weights[11])
        if self.max_pool:
            h2 = F.max_pool2d(h2, kernel_size=2)
        h2 = F.relu(h2)

        h3 = F.conv2d(h2, weight=weights[4], bias=weights[5], stride=1, padding=1)
        h3 = F.batch_norm(h3, running_mean=self.bn3.running_mean, running_var=self.bn3.running_var, weight=weights[12], bias=weights[13])
        if self.max_pool:
            h3 = F.max_pool2d(h3, kernel_size=2)
        h3 = F.relu(h3)

        h4 = F.conv2d(h3, weight=weights[6], bias=weights[7], stride=1, padding=1)
        h4 = F.batch_norm(h4, running_mean=self.bn4.running_mean, running_var=self.bn4.running_var, weight=weights[14], bias=weights[15])
        if self.max_pool:
            h4 = F.max_pool2d(h4, kernel_size=2)
        h4 = F.relu(h4)

        h4 = h4.view(h4.size(0), -1)

        y = F.linear(h4, weights[-2], weights[-1])
        return y


class CAVIAModel(nn.Module):
    def __init__(self, n_input=1, n_output=1, n_hidden=[40, 40], num_context_params=4, device="cpu"):
        super(CAVIAModel, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.num_context_params = num_context_params
        self.device = device
        self.layer = nn.ModuleList()

        self.layer.append(nn.Linear(n_input + num_context_params, n_hidden[0], bias=True))
        self.layer.append(nn.ReLU())
        for i in range(len(n_hidden) - 1):
            self.layer.append(nn.Linear(n_hidden[i], n_hidden[i + 1], bias=True))
            self.layer.append(nn.ReLU())
        self.layer.append(nn.Linear(n_hidden[-1], n_output, bias=True))

        self.context_params = None

        self.reset_params()
        self.reset_context_params()
        self.directory = "Sinusoid"

    def reset_params(self):
        for i in range(0, len(self.layer) + 1, 2):
            nn.init.trunc_normal_(self.layer[i].weight, mean=0, std=0.01, a=-0.02, b=0.02)
            nn.init.zeros_(self.layer[i].bias)

    def reset_context_params(self):
        self.context_params = torch.zeros(self.num_context_params).to(self.device)
        self.context_params.requires_grad = True

    def forward(self, x):
        x = torch.cat((x, self.context_params.expand(x.shape[0], -1)), dim=1)
        for i in range(0, len(self.layer) - 1, 2):
            x = self.layer[i](x)
            x = F.relu(x)
        x = self.layer[-1](x)

        return x

    def parameterised(self, x, weights):
        x = torch.cat((x, self.context_params.expand(x.shape[0], -1)), dim=1)

        for i in range(0, len(weights) - 2, 2):
            x = F.linear(x, weights[i], weights[i + 1])
            x = F.relu(x)
        x = F.linear(x, weights[-2], weights[-1])

        return x


class CAVIAConvModel(nn.Module):
    def __init__(self, num_classes, num_filters, max_pool, num_context_params, context_in, num_film_hidden_layers,
                 imsize, initialisation="kaiming", device="cpu"):
        super(CAVIAConvModel, self).__init__()

        self.num_classes = num_classes
        self.num_filters = num_filters
        self.max_pool = max_pool
        self.num_context_params = num_context_params
        self.context_in = context_in
        self.num_film_hidden_layers = num_film_hidden_layers
        self.kernel_size = 3

        stride = 1
        padding = 1
        self.num_channels = 3

        self.conv1 = nn.Conv2d(self.num_channels, self.num_filters, self.kernel_size, stride=stride,
                               padding=padding).to(device)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride, padding=padding).to(
            device)
        self.conv3 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride, padding=padding).to(
            device)
        if not self.max_pool:
            self.conv4 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride).to(device)
        else:
            self.conv4 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride,
                                   padding=padding).to(device)

        # # batch norm
        self.bn1 = nn.BatchNorm2d(self.num_filters, track_running_stats=False).to(device)
        self.bn2 = nn.BatchNorm2d(self.num_filters, track_running_stats=False).to(device)
        self.bn3 = nn.BatchNorm2d(self.num_filters, track_running_stats=False).to(device)
        self.bn4 = nn.BatchNorm2d(self.num_filters, track_running_stats=False).to(device)

        # initialise weights for the fully connected layer
        if imsize == 84:
            self.fc1 = nn.Linear(5 * 5 * self.num_filters + int(context_in[4]) * num_context_params,
                                 self.num_classes).to(device)
        elif imsize == 28:
            self.fc1 = nn.Linear(self.num_filters + int(context_in[4]) * num_context_params, self.num_classes).to(
                device)
        else:
            raise NotImplementedError('Cannot handle image size.')

        # -- additions to enable context parameters at convolutional layers --

        # for each layer where we have context parameters, initialise a FiLM layer
        if self.context_in[0]:
            self.film1 = nn.Linear(self.num_context_params, self.num_filters * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film11 = nn.Linear(self.num_filters * 2, self.num_filters * 2).to(device)
        if self.context_in[1]:
            self.film2 = nn.Linear(self.num_context_params, self.num_filters * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film22 = nn.Linear(self.num_filters * 2, self.num_filters * 2).to(device)
        if self.context_in[2]:
            self.film3 = nn.Linear(self.num_context_params, self.num_filters * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film33 = nn.Linear(self.num_filters * 2, self.num_filters * 2).to(device)
        if self.context_in[3]:
            self.film4 = nn.Linear(self.num_context_params, self.num_filters * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film44 = nn.Linear(self.num_filters * 2, self.num_filters * 2).to(device)

        # parameter initialisation (if different than standard pytorch one)
        if initialisation != 'standard':
            self.init_params(initialisation)

        # initialise context parameters
        self.context_params = torch.zeros(size=[self.num_context_params], requires_grad=True).to(device)

        self.directory = "MiniImageNet"

    def init_params(self, initialisation):
        # convolutional weights
        if initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu', self.conv1.weight))
            torch.nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu', self.conv2.weight))
            torch.nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu', self.conv3.weight))
            torch.nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('relu', self.conv4.weight))
        elif initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')

        # convolutional bias
        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)
        self.conv3.bias.data.fill_(0)
        self.conv4.bias.data.fill_(0)

        # fully connected weights at the end
        if initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('linear', self.fc1.weight))
        elif initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='linear')

        # fully connected bias
        self.fc1.bias.data.fill_(0)

        # FiLM layer weights
        if self.context_in[0] and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film1.weight, gain=nn.init.calculate_gain('linear', self.film1.weight))
        elif self.context_in[0] and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film1.weight, nonlinearity='linear')

        if self.context_in[1] and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film2.weight, gain=nn.init.calculate_gain('linear', self.film2.weight))
        elif self.context_in[1] and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film2.weight, nonlinearity='linear')

        if self.context_in[2] and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film3.weight, gain=nn.init.calculate_gain('linear', self.film3.weight))
        elif self.context_in[2] and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film3.weight, nonlinearity='linear')

        if self.context_in[3] and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film4.weight, gain=nn.init.calculate_gain('linear', self.film4.weight))
        elif self.context_in[3] and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film4.weight, nonlinearity='linear')

    def reset_context_params(self):
        self.context_params = self.context_params.detach() * 0
        self.context_params.requires_grad = True

    def forward(self, x):
        # pass through convolutional layer
        h1 = self.conv1(x)
        # batchnorm
        h1 = self.bn1(h1)
        # do max-pooling (for imagenet)
        if self.max_pool:
            h1 = F.max_pool2d(h1, kernel_size=2)
        # if we have context parameters, adjust conv output using FiLM variables
        if self.context_in[0]:
            # FiLM it: forward through film layer to get scale and shift parameter
            film1 = self.film1(self.context_params)
            if self.num_film_hidden_layers == 1:
                film1 = self.film11(F.relu(film1))
            gamma1 = film1[:self.num_filters].view(1, -1, 1, 1)
            beta1 = film1[self.num_filters:].view(1, -1, 1, 1)
            # transform feature map
            h1 = gamma1 * h1 + beta1
        # pass through ReLu activation function
        h1 = F.relu(h1)

        h2 = self.conv2(h1)
        h2 = self.bn2(h2)
        if self.max_pool:
            h2 = F.max_pool2d(h2, kernel_size=2)
        if self.context_in[1]:
            film2 = self.film2(self.context_params)
            if self.num_film_hidden_layers == 1:
                film2 = self.film22(F.relu(film2))
            gamma2 = film2[:self.num_filters].view(1, -1, 1, 1)
            beta2 = film2[self.num_filters:].view(1, -1, 1, 1)
            h2 = gamma2 * h2 + beta2
        h2 = F.relu(h2)

        h3 = self.conv3(h2)
        h3 = self.bn3(h3)
        if self.max_pool:
            h3 = F.max_pool2d(h3, kernel_size=2)
        if self.context_in[2]:
            film3 = self.film3(self.context_params)
            if self.num_film_hidden_layers == 1:
                film3 = self.film33(F.relu(film3))
            gamma3 = film3[:self.num_filters].view(1, -1, 1, 1)
            beta3 = film3[self.num_filters:].view(1, -1, 1, 1)
            h3 = gamma3 * h3 + beta3
        h3 = F.relu(h3)

        h4 = self.conv4(h3)
        h4 = self.bn4(h4)
        if self.max_pool:
            h4 = F.max_pool2d(h4, kernel_size=2)
        if self.context_in[3]:
            film4 = self.film4(self.context_params)
            if self.num_film_hidden_layers == 1:
                film4 = self.film44(F.relu(film4))
            gamma4 = film4[:self.num_filters].view(1, -1, 1, 1)
            beta4 = film4[self.num_filters:].view(1, -1, 1, 1)
            h4 = gamma4 * h4 + beta4
        h4 = F.relu(h4)

        # flatten
        h4 = h4.view(h4.size(0), -1)

        if self.context_in[4]:
            h4 = torch.cat((h4, self.context_params.expand(h4.size(0), -1)), dim=1)

        y = self.fc1(h4)

        return y


class MAML():
    def __init__(self, args, model, tasks, num_tasks_mtrain):
        self.args = args
        # Model
        self.model = model
        self.penalty_type = args.penalty_type

        # Model hyperparameter
        self.meta_lr = args.meta_lr
        self.inner_lr = args.inner_lr
        self.meta_batch_size = args.meta_batch_size
        self.update_batch_size = args.update_batch_size
        self.step = args.step

        # Datasource
        self.datasource = args.datasource
        self.tasks = tasks
        self.num_tasks_mtrain = num_tasks_mtrain

        # Optimizer
        self.weights = list(self.model.parameters())
        self.meta_optimiser = torch.optim.Adam(self.weights, lr=self.meta_lr)
        if self.datasource == 1:
            self.criterion = nn.MSELoss()
        elif self.datasource == 2:
            self.criterion = F.cross_entropy
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimiser, 5000, args.lr_meta_decay)

        # Log
        if self.datasource == 1:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta}, {inner})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size,
                task=self.update_batch_size, meta=self.meta_lr, inner=self.inner_lr)
        elif self.datasource == 2:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta}, {inner})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size, n_way=args.n_way,
                k_shot=args.k_shot, k_query=args.k_query, meta=self.meta_lr, inner=self.inner_lr)

        self.plot_every = args.plot_log
        self.print_interval = args.print_log
        self.save_interval = args.save_log
        self.meta_losses = []

        # Penalty hyperparameter
        if self.penalty_type == 1:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight1 * (self.meta_batch_size * self.meta_batch_size / self.inner_lr / self.inner_lr / self.num_tasks_mtrain / self.num_tasks_mtrain)
            if self.datasource == 1:
                self.avg = [torch.zeros(w.size(), requires_grad=False) for w in self.weights]
            elif self.datasource == 2:
                self.avg = [torch.zeros(w.size(), requires_grad=False).to(self.args.device) for w in self.weights]

        if self.penalty_type == 2 or self.penalty_type == 3:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight2 * self.meta_batch_size
            self.default = args.weight3
            if self.datasource == 1:
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)
            elif self.datasource == 2:
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

    def run_experiment(self, num_iterations):
        epoch_loss = 0
        self.model.train()

        if self.datasource == 1:
            for iteration in range(1, num_iterations + 1):
                self.meta_optimiser.zero_grad()

                meta_loss = 0
                losses, index = [], []

                # inner-loop
                for _ in range(self.meta_batch_size):
                    idx, task = self.tasks.sample_task(train=True)

                    task_specific_parameter = [w.clone() for w in self.weights]
                    training_inputs, training_outputs = task.sample_data(K=self.update_batch_size)

                    # Compute training loss of current task
                    training_loss = self.criterion(self.model.parameterised(training_inputs, task_specific_parameter), training_outputs)

                    if self.penalty_type == 0:
                        pass
                    elif self.penalty_type == 1:
                        for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter):
                            training_loss = training_loss + self.penalty_lr * self.penalty_criterion(w1 + w2 / self.num_tasks_mtrain, w3 / self.num_tasks_mtrain)
                    elif self.penalty_type == 2:
                        penalty_numerator = 0
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            penalty_numerator += self.penalty_criterion(w1, w2)
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    elif self.penalty_type == 3:
                        penalty_numerator = 0
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            penalty_numerator += self.penalty_criterion(w1, w2) ** 2
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    else:
                        raise RuntimeError('Invalid penalty-type index.')

                    # Computing task-specific parameter
                    grad = torch.autograd.grad(training_loss, task_specific_parameter, create_graph=True, retain_graph=True)
                    task_specific_parameter = [w - self.inner_lr * g for w, g in zip(task_specific_parameter, grad)]

                    # Update penalty parameters
                    if self.penalty_type == 1:
                        self.avg = [w1 + w2.detach().clone() / self.num_tasks_mtrain - w3.detach().clone() / self.num_tasks_mtrain for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter)]

                    if self.penalty_type == 2:
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone()

                    if self.penalty_type == 3:
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone() ** 2

                    # Compute validation loss of current task
                    val_inputs, val_outputs = task.sample_data(K=self.update_batch_size)
                    val_loss = self.criterion(self.model.parameterised(val_inputs, task_specific_parameter), val_outputs)

                    losses.append(val_loss)
                    index.append(idx)

                # Compute meta-loss
                for i in range(self.meta_batch_size):
                    meta_loss += losses[i]

                # Compute meta-gradient on meta-loss
                meta_grads = torch.autograd.grad(meta_loss, self.weights)
                with torch.no_grad():
                    for w, g in zip(self.weights, meta_grads):
                        w.grad = g

                self.meta_optimiser.step()

                # Reset penalty parameters
                if self.penalty_type == 1:
                    self.avg = [torch.zeros(w.size(), requires_grad=False) for w in self.weights]

                if self.penalty_type == 2 or self.penalty_type == 3:
                    self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)

                # Summaries
                epoch_loss += meta_loss.item()
                if iteration % self.print_interval == 0:
                    print("{iter}/{total_iter}. loss: {loss}".format(iter=iteration, total_iter=num_iterations, loss=epoch_loss / self.plot_every / self.meta_batch_size))

                if iteration % self.plot_every == 0:
                    self.meta_losses.append(epoch_loss / self.plot_every)
                    epoch_loss = 0

                if iteration % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))

        elif self.datasource == 2:
            iteration = 1

            if self.args.resume:
                iteration = self.args.save_iter
                init_model = torch.load(self.directory + "_iter = {iter}.pt".format(iter = iteration), map_location=torch.device(self.args.device))
                self.model.load_state_dict(init_model)

            while iteration < num_iterations + 1:
                tasks = self.tasks(mode='train', n_way=self.args.n_way, k_shot=self.args.k_shot, k_query=self.args.k_query, batchsz=10000, imsize=self.args.imsize, data_path=self.args.data_path)
                dataloader = DataLoader(tasks, self.meta_batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=False)

                for step, batch in enumerate(dataloader):
                    self.scheduler.step()

                    training_inputs = batch[0].to(self.args.device)
                    training_outputs = batch[1].to(self.args.device)
                    val_inputs = batch[2].to(self.args.device)
                    val_outputs = batch[3].to(self.args.device)

                    if training_inputs.shape[0] != self.meta_batch_size:
                        continue

                    self.meta_optimiser.zero_grad()

                    meta_loss = 0
                    losses = []

                    task_specific_parameter_list = [[w.clone() for w in self.weights] for _ in range(self.meta_batch_size)]

                    # inner-loop
                    for _ in range(self.step):
                        for i in range(self.meta_batch_size):
                            task_specific_parameter = task_specific_parameter_list[i]
                            # Compute training loss of current task
                            training_loss = self.criterion(self.model.parameterised(training_inputs[i], task_specific_parameter), training_outputs[i])

                            if self.penalty_type == 0:
                                pass
                            elif self.penalty_type == 1:
                                for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter):
                                    training_loss = training_loss + self.penalty_lr * self.penalty_criterion(
                                        w1 + w2 / self.num_tasks_mtrain, w3 / self.num_tasks_mtrain)
                            elif self.penalty_type == 2:
                                penalty_numerator = 0
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    penalty_numerator += self.penalty_criterion(w1, w2)
                                penalty_denominator = self.denominator + penalty_numerator
                                training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                            elif self.penalty_type == 3:
                                penalty_numerator = 0
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    penalty_numerator += self.penalty_criterion(w1, w2) ** 2
                                penalty_denominator = self.denominator + penalty_numerator
                                training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                            else:
                                raise RuntimeError('Invalid penalty-type index.')

                            # Computing task-specific parameter
                            grad = torch.autograd.grad(training_loss, task_specific_parameter, create_graph=True, retain_graph=True)
                            task_specific_parameter = [w - self.inner_lr * g for w, g in zip(task_specific_parameter, grad)]

                            task_specific_parameter_list[i] = [w.clone() for w in task_specific_parameter]

                            # Update penalty parameters
                            if self.penalty_type == 1:
                                self.avg = [w1 + w2.detach().clone() / self.num_tasks_mtrain - w3.detach().clone() / self.num_tasks_mtrain for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter)]

                            if self.penalty_type == 2:
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone()

                            if self.penalty_type == 3:
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone() ** 2


                    for i in range(self.meta_batch_size):
                        # Compute validation loss of current task
                        val_loss = self.criterion(self.model.parameterised(val_inputs[i], task_specific_parameter_list[i]), val_outputs[i])

                        losses.append(val_loss)

                    # Compute meta-loss
                    for i in range(self.meta_batch_size):
                        meta_loss += losses[i]
                    meta_loss = meta_loss / self.meta_batch_size

                    # Compute meta-gradient
                    meta_grads = torch.autograd.grad(meta_loss, self.weights)
                    with torch.no_grad():
                        for w, g in zip(self.weights, meta_grads):
                            w.grad = g.clamp(-10, 10)

                    self.meta_optimiser.step()

                    # Reset penalty parameters
                    if self.penalty_type == 1:
                        self.avg = [torch.zeros(w.size(), requires_grad=False).to(self.args.device) for w in self.weights]

                    if self.penalty_type == 2 or self.penalty_type == 3:
                        self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

                    # Summaries
                    epoch_loss += meta_loss
                    if iteration % self.print_interval == 0:
                        print("{iter}/{total_iter}. loss: {loss}".format(iter=iteration, total_iter=num_iterations, loss=epoch_loss / self.plot_every / self.meta_batch_size))

                    if iteration % self.plot_every == 0:
                        self.meta_losses.append(epoch_loss / self.plot_every)
                        epoch_loss = 0

                    if iteration % self.save_interval == 0:
                        torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))

                    iteration += 1

                    if iteration > num_iterations + 1:
                        break


class TRMAML():
    def __init__(self, args, model, tasks, num_tasks_mtrain):
        self.args = args
        # Model
        self.model = model
        self.penalty_type = args.penalty_type

        # Model hyperparameter
        self.meta_lr = args.meta_lr
        self.inner_lr = args.inner_lr
        self.p_lr = args.p_lr
        self.meta_batch_size = args.meta_batch_size
        self.update_batch_size = args.update_batch_size
        self.step = args.step

        # Datasource
        self.datasource = args.datasource
        self.tasks = tasks
        self.num_tasks_mtrain = num_tasks_mtrain

        # Optimizer
        self.weights = list(self.model.parameters())
        self.meta_optimiser = torch.optim.Adam(self.weights, lr=self.meta_lr)
        if self.datasource == 1:
            self.criterion = nn.MSELoss()
        elif self.datasource == 2:
            self.criterion = F.cross_entropy
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimiser, 5000, args.lr_meta_decay)

        # Log
        if self.datasource == 1:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta}, {inner}, {prob})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size,
                task=self.update_batch_size, meta=self.meta_lr, inner=self.inner_lr, prob=self.p_lr)
        elif self.datasource == 2:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta}, {inner}, {prob})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size, n_way=args.n_way,
                k_shot=args.k_shot, k_query=args.k_query, meta=self.meta_lr, inner=self.inner_lr, prob=self.p_lr)

        self.plot_every = args.plot_log
        self.print_interval = args.print_log
        self.save_interval = args.save_log
        self.meta_losses = []

        # Penalty hyperparameter
        if self.penalty_type == 1:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight1 * (self.meta_batch_size * self.meta_batch_size / self.inner_lr / self.inner_lr / self.num_tasks_mtrain / self.num_tasks_mtrain)
            if self.datasource == 1:
                self.avg = [torch.zeros(w.size(), requires_grad=False) for w in self.weights]
            elif self.datasource == 2:
                self.avg = [torch.zeros(w.size(), requires_grad=False).to(self.args.device) for w in self.weights]

        if self.penalty_type == 2 or self.penalty_type == 3:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight2 * self.meta_batch_size
            self.default = args.weight3
            if self.datasource == 1:
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)
            elif self.datasource == 2:
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

    def run_experiment(self, num_iterations):
        epoch_loss = 0
        p = np.ones(self.num_tasks_mtrain)
        self.model.train()

        if self.datasource == 1:
            for iteration in range(1, num_iterations + 1):
                self.meta_optimiser.zero_grad()
                meta_loss = 0
                losses, index = [], []
                p = torch.tensor(p, dtype=torch.float32, requires_grad=False)

                # inner-loop
                for i in range(self.meta_batch_size):
                    idx, task = self.tasks.sample_task(train=True)
                    task_specific_parameter = [w.clone() for w in self.weights]
                    training_inputs, training_outputs = task.sample_data(K=self.update_batch_size)

                    # Compute training loss of current task
                    training_loss = self.criterion(self.model.parameterised(training_inputs, task_specific_parameter), training_outputs)

                    if self.penalty_type == 0:
                        pass
                    elif self.penalty_type == 1:
                        for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter):
                            training_loss = training_loss + self.penalty_lr * self.penalty_criterion(w1 + w2 / self.num_tasks_mtrain, w3 / self.num_tasks_mtrain)
                    elif self.penalty_type == 2:
                        penalty_numerator = 0
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            penalty_numerator += self.penalty_criterion(w1, w2)
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    elif self.penalty_type == 3:
                        penalty_numerator = 0
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            penalty_numerator += self.penalty_criterion(w1, w2) ** 2
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    else:
                        raise RuntimeError('Invalid penalty-type index.')

                    # Computing task-specific parameter
                    grad = torch.autograd.grad(training_loss, task_specific_parameter, create_graph=True, retain_graph=True)
                    task_specific_parameter = [w - self.inner_lr * g for w, g in zip(task_specific_parameter, grad)]

                    # Update penalty parameters
                    if self.penalty_type == 1:
                        self.avg = [w1 + w2.detach().clone() / self.num_tasks_mtrain - w3.detach().clone() / self.num_tasks_mtrain for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter)]

                    if self.penalty_type == 2:
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone()

                    if self.penalty_type == 3:
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone() ** 2

                    # Compute validation loss of current task
                    val_inputs, val_outputs = task.sample_data(K=self.update_batch_size)
                    val_loss = self.criterion(self.model.parameterised(val_inputs, task_specific_parameter), val_outputs)

                    losses.append(val_loss)
                    index.append(idx)

                # Compute meta-loss
                for i in range(self.meta_batch_size):
                    meta_loss += p[index[i]] * losses[i]

                # Compute meta-gradient on meta-loss
                meta_grads = torch.autograd.grad(meta_loss, self.weights)
                with torch.no_grad():
                    for w, g in zip(self.weights, meta_grads):
                        w.grad = g

                self.meta_optimiser.step()

                # Compute probability of task
                real_p = p / self.num_tasks_mtrain
                for i in range(self.meta_batch_size):
                    real_p[index[i]] += self.p_lr * losses[i].item()
                p = self.num_tasks_mtrain * simplex_proj(real_p.numpy())

                # Reset penalty parameters
                if self.penalty_type == 1:
                    self.avg = [torch.zeros(w.size(), requires_grad=False) for w in self.weights]

                if self.penalty_type == 2 or self.penalty_type == 3:
                    self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)

                # Summaries
                epoch_loss += meta_loss.item()
                if iteration % self.print_interval == 0:
                    print("{iter}/{total_iter}. loss: {loss}".format(iter=iteration, total_iter=num_iterations, loss=epoch_loss / self.plot_every / self.meta_batch_size))

                if iteration % self.plot_every == 0:
                    self.meta_losses.append(epoch_loss / self.plot_every)
                    epoch_loss = 0

                if iteration % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))

        elif self.datasource == 2:
            raise RuntimeError('Invalid datasource for TRMAML.')


class Reptile():
    def __init__(self, args, model, tasks, num_tasks_mtrain):
        self.args = args
        # Model
        self.model = model
        self.penalty_type = args.penalty_type

        # Model hyperparameter
        self.meta_lr = args.epsilon / args.inner_lr
        self.inner_lr = args.inner_lr
        self.meta_batch_size = args.meta_batch_size
        self.update_batch_size = args.update_batch_size
        self.step = args.step

        # Datasource
        self.datasource = args.datasource
        self.tasks = tasks
        self.num_tasks_mtrain = num_tasks_mtrain

        # Optimizer
        self.weights = list(self.model.parameters())
        if self.datasource == 1:
            self.criterion = nn.MSELoss()
        elif self.datasource == 2:
            self.criterion = F.cross_entropy

        # Log
        if self.datasource == 1:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta}, {inner})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size,
                task=self.update_batch_size, meta=args.epsilon, inner=self.inner_lr)
        elif self.datasource == 2:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta}, {inner})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size, n_way=args.n_way,
                k_shot=args.k_shot, k_query=args.k_query, meta=args.epsilon, inner=self.inner_lr)

        self.plot_every = args.plot_log
        self.print_interval = args.print_log
        self.save_interval = args.save_log
        self.meta_losses = []

        # Penalty hyperparameter
        if self.penalty_type == 1:
            self.dim_weights = 0.0
            for w in self.weights:
                self.dim_weights += len(w)
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight1 * (self.num_tasks_mtrain * self.num_tasks_mtrain / self.dim_weights / self.inner_lr / self.inner_lr / self.meta_batch_size / self.meta_batch_size)
            if self.datasource == 1:
                self.avg = [torch.zeros(w.size(), requires_grad=False) for w in self.weights]
            elif self.datasource == 2:
                self.avg = [torch.zeros(w.size(), requires_grad=False).to(self.args.device) for w in self.weights]

        if self.penalty_type == 2 or self.penalty_type == 3:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight2 * self.meta_batch_size
            self.default = args.weight3
            if self.datasource == 1:
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)
            elif self.datasource == 2:
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

    def run_experiment(self, num_iterations):
        epoch_loss = 0
        self.model.train()

        if self.datasource == 1:
            for iteration in range(1, num_iterations + 1):
                self.model.zero_grad()
                meta_loss = 0
                losses, index, gradient_list = [], [], []

                # inner-loop
                for i in range(self.meta_batch_size):
                    idx, task = self.tasks.sample_task(train=True)
                    task_specific_parameter = [w.clone() for w in self.weights]
                    training_inputs, training_outputs = task.sample_data(K=self.update_batch_size)

                    # Compute training loss of current task
                    training_loss = self.criterion(self.model.parameterised(training_inputs, task_specific_parameter), training_outputs)

                    if self.penalty_type == 0:
                        pass
                    elif self.penalty_type == 1:
                        for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter):
                            training_loss = training_loss + self.penalty_lr * self.penalty_criterion(w1 + w2 / self.num_tasks_mtrain, w3 / self.num_tasks_mtrain)
                    elif self.penalty_type == 2:
                        penalty_numerator = 0
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            penalty_numerator += self.penalty_criterion(w1, w2)
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    elif self.penalty_type == 3:
                        penalty_numerator = 0
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            penalty_numerator += self.penalty_criterion(w1, w2) ** 2
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    else:
                        raise RuntimeError('Invalid penalty-type index.')

                    # Computing task-specific parameter
                    grad = torch.autograd.grad(training_loss, task_specific_parameter, requires_grad=True, retain_graph=True)
                    task_specific_parameter = [w - self.inner_lr * g for w, g in zip(task_specific_parameter, grad)]
                    task_specific_grad = [self.inner_lr * g.detach().clone() for g in grad]

                    # Update penalty parameters
                    if self.penalty_type == 1:
                        self.avg = [w1 + w2.detach().clone() / self.num_tasks_mtrain - w3.detach().clone() / self.num_tasks_mtrain for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter)]

                    if self.penalty_type == 2:
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone()

                    if self.penalty_type == 3:
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone() ** 2

                    # Compute validation loss of current task
                    val_inputs, val_outputs = task.sample_data(K=self.update_batch_size)
                    val_loss = self.criterion(self.model.parameterised(val_inputs, task_specific_parameter), val_outputs)

                    losses.append(val_loss)
                    index.append(idx)
                    gradient_list.append(task_specific_grad)

                # Compute meta-loss and update meta-parameter
                for i in range(self.meta_batch_size):
                    self.weights = [w1 - self.args.epsilon * w2 / self.meta_batch_size for w1, w2 in zip(self.weights, gradient_list[i])]
                    meta_loss += losses[i]

                # Reset penalty parameters
                if self.penalty_type == 1:
                    self.avg = [torch.zeros(w.size(), requires_grad=False) for w in self.weights]

                if self.penalty_type == 2 or self.penalty_type == 3:
                    self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)

                # Summaries
                epoch_loss += meta_loss.item()
                if iteration % self.print_interval == 0:
                    print("{iter}/{total_iter}. loss: {loss}".format(iter=iteration, total_iter=num_iterations, loss=epoch_loss / self.plot_every / self.meta_batch_size))

                if iteration % self.plot_every == 0:
                    self.meta_losses.append(epoch_loss / self.plot_every)
                    epoch_loss = 0

                if iteration % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))

        elif self.datasource == 2:
            iteration = 1

            if self.args.resume:
                iteration = self.args.save_iter
                init_model = torch.load(self.directory + "_iter = {iter}.pt".format(iter = iteration), map_location=torch.device(self.args.device))
                self.model.load_state_dict(init_model)

            while iteration < num_iterations + 1:
                tasks = self.tasks(mode='train', n_way=self.args.n_way, k_shot=self.args.k_shot,
                                   k_query=self.args.k_query, batchsz=10000, imsize=self.args.imsize,
                                   data_path=self.args.data_path)
                dataloader = DataLoader(tasks, self.meta_batch_size, shuffle=True, num_workers=self.args.num_workers,
                                        pin_memory=False)

                for step, batch in enumerate(dataloader):
                    self.scheduler.step()

                    training_inputs = batch[0].to(self.args.device)
                    training_outputs = batch[1].to(self.args.device)

                    if training_inputs.shape[0] != self.meta_batch_size:
                        continue

                    self.model.zero_grad()

                    task_specific_parameter_list = [[w.clone() for w in self.weights] for _ in range(self.meta_batch_size)]

                    # inner-loop
                    for _ in range(self.step):
                        for i in range(self.meta_batch_size):
                            task_specific_parameter = task_specific_parameter_list[i]
                            # Compute training loss of current task
                            training_loss = self.criterion(self.model.parameterised(training_inputs[i], task_specific_parameter), training_outputs[i])

                            if self.penalty_type == 0:
                                pass
                            elif self.penalty_type == 1:
                                for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter):
                                    training_loss = training_loss + self.penalty_lr * self.penalty_criterion(w1 + w2 / self.num_tasks_mtrain, w3 / self.num_tasks_mtrain)
                            elif self.penalty_type == 2:
                                penalty_numerator = 0
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    penalty_numerator += self.penalty_criterion(w1, w2)
                                penalty_denominator = self.denominator + penalty_numerator
                                training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                            elif self.penalty_type == 3:
                                penalty_numerator = 0
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    penalty_numerator += self.penalty_criterion(w1, w2) ** 2
                                penalty_denominator = self.denominator + penalty_numerator
                                training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                            else:
                                raise RuntimeError('Invalid penalty-type index.')

                            # Computing task-specific parameter
                            grad = torch.autograd.grad(training_loss, task_specific_parameter, create_graph=True, retain_graph=True)
                            task_specific_parameter = [w - self.inner_lr * g for w, g in zip(task_specific_parameter, grad)]

                            task_specific_parameter_list[i] = [w.clone() for w in task_specific_parameter]

                            # Update penalty parameters
                            if self.penalty_type == 1:
                                self.avg = [w1 + w2.detach().clone() / self.num_tasks_mtrain - w3.detach().clone() / self.num_tasks_mtrain for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter)]

                            if self.penalty_type == 2:
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone()

                            if self.penalty_type == 3:
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone() ** 2

                    # Update meta-parameter
                    for i in range(self.meta_batch_size):
                        self.weights = [w1 + self.meta_lr * (w2 - w1) / self.meta_batch_size for w1, w2 in zip(self.weights, task_specific_parameter_list[i])]

                    # Reset penalty parameters
                    if self.penalty_type == 1:
                        self.avg = [torch.zeros(w.size(), requires_grad=False).to(self.args.device) for w in self.weights]

                    if self.penalty_type == 2 or self.penalty_type == 3:
                        self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

                    # Summaries
                    if iteration % self.save_interval == 0:
                        torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))

                    iteration += 1

                    if iteration > num_iterations + 1:
                        break


class MetaSGD():
    def __init__(self, args, model, tasks, num_tasks_mtrain):
        self.args = args
        # Model
        self.model = model
        self.penalty_type = args.penalty_type

        # Model hyperparameter
        self.meta_lr = args.meta_lr
        if args.datasource == 1:
            self.inner_lr = [torch.rand(w.size(), requires_grad=True) * 0.0095 + 0.0005 for w in list(self.model.parameters())]
        elif args.datasource == 2:
            self.inner_lr = [torch.rand(w.size(), requires_grad=True).to(self.args.device) * 0.0095 + 0.0005 for w in list(self.model.parameters())]
        self.meta_batch_size = args.meta_batch_size
        self.update_batch_size = args.update_batch_size
        self.step = args.step

        # Datasource
        self.datasource = args.datasource
        self.tasks = tasks
        self.num_tasks_mtrain = num_tasks_mtrain

        # Optimizer
        self.weights = list(self.model.parameters())
        self.meta_optimiser = torch.optim.Adam(self.weights, lr=self.meta_lr)
        if self.datasource == 1:
            self.criterion = nn.MSELoss()
        elif self.datasource == 2:
            self.criterion = F.cross_entropy
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimiser, 5000, args.lr_meta_decay)

        # Log
        if self.datasource == 1:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size,
                task=self.update_batch_size, meta=self.meta_lr)
        elif self.datasource == 2:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size, n_way=args.n_way,
                k_shot=args.k_shot, k_query=args.k_query, meta=self.meta_lr)

        self.plot_every = args.plot_log
        self.print_interval = args.print_log
        self.save_interval = args.save_log
        self.meta_losses = []

        # Penalty hyperparameter
        if self.penalty_type == 1:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight1 * (self.meta_batch_size * self.meta_batch_size / 0.05 / 0.05 / self.num_tasks_mtrain / self.num_tasks_mtrain)
            if self.datasource == 1:
                self.avg = [torch.zeros(w.size(), requires_grad=False) for w in self.weights]
            elif self.datasource == 2:
                self.avg = [torch.zeros(w.size(), requires_grad=False).to(self.args.device) for w in self.weights]

        if self.penalty_type == 2 or self.penalty_type == 3:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight2 * self.meta_batch_size
            self.default = args.weight3
            if self.datasource == 1:
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)
            elif self.datasource == 2:
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

    def run_experiment(self, num_iterations):
        epoch_loss = 0
        self.model.train()

        if self.datasource == 1:
            for iteration in range(1, num_iterations + 1):
                self.meta_optimiser.zero_grad()
                meta_loss = 0
                losses, index = [], []

                # inner-loop
                for i in range(self.meta_batch_size):
                    idx, task = self.tasks.sample_task(train=True)

                    task_specific_parameter = [w.clone() for w in self.weights]
                    training_inputs, training_outputs = task.sample_data(K=self.update_batch_size)

                    # Compute training loss of current task
                    training_loss = self.criterion(self.model.parameterised(training_inputs, task_specific_parameter), training_outputs)

                    if self.penalty_type == 0:
                        pass
                    elif self.penalty_type == 1:
                        for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter):
                            training_loss = training_loss + self.penalty_lr * self.penalty_criterion(w1 + w2 / self.num_tasks_mtrain, w3 / self.num_tasks_mtrain)
                    elif self.penalty_type == 2:
                        penalty_numerator = 0
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            penalty_numerator += self.penalty_criterion(w1, w2)
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    elif self.penalty_type == 3:
                        penalty_numerator = 0
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            penalty_numerator += self.penalty_criterion(w1, w2) ** 2
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    else:
                        raise RuntimeError('Invalid penalty-type index.')

                    # Computing task-specific parameter
                    grad = torch.autograd.grad(training_loss, task_specific_parameter, create_graph=True, retain_graph=True)
                    task_specific_parameter = [w - torch.mul(a, g) for w, a, g in zip(task_specific_parameter, self.inner_lr, grad)]

                    # Update penalty parameters
                    if self.penalty_type == 1:
                        self.avg = [w1 + w2.detach().clone() / self.num_tasks_mtrain - w3.detach().clone() / self.num_tasks_mtrain for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter)]

                    if self.penalty_type == 2:
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone()

                    if self.penalty_type == 3:
                        for w1, w2 in zip(self.weights, task_specific_parameter):
                            self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone() ** 2

                    # Compute validation loss of current task
                    val_inputs, val_outputs = task.sample_data(K=self.update_batch_size)
                    val_loss = self.criterion(self.model.parameterised(val_inputs, task_specific_parameter), val_outputs)

                    losses.append(val_loss)
                    index.append(idx)

                # Compute meta-loss
                for i in range(self.meta_batch_size):
                    meta_loss += losses[i]

                # Compute meta-gradient on meta-loss
                meta_grads = torch.autograd.grad(meta_loss, self.weights + self.inner_lr)
                with torch.no_grad():
                    for w, g in zip(self.weights + self.inner_lr, meta_grads):
                        w.grad = g

                # self.inner_lr = [torch.tensor(torch.clamp(w - self.meta_lr * w.grad, min=0, max=0.05), requires_grad=True) for w in self.inner_lr]
                self.inner_lr = [torch.tensor(w - self.meta_lr * w.grad, requires_grad=True) for w in self.inner_lr]

                self.meta_optimiser.step()

                # Reset penalty parameters
                if self.penalty_type == 1:
                    self.avg = [torch.zeros(w.size(), requires_grad=False) for w in self.weights]

                if self.penalty_type == 2 or self.penalty_type == 3:
                    self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)

                # Summaries
                epoch_loss += meta_loss.item()
                if iteration % self.print_interval == 0:
                    print("{iter}/{total_iter}. loss: {loss}".format(iter=iteration, total_iter=num_iterations, loss=epoch_loss / self.plot_every / self.meta_batch_size))

                if iteration % self.plot_every == 0:
                    self.meta_losses.append(epoch_loss / self.plot_every)
                    epoch_loss = 0

                if iteration % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))
                    torch.save(self.inner_lr, self.directory + "_lr_iter = {iter}.pt".format(iter=iteration))

        elif self.datasource == 2:
            iteration = 1

            if self.args.resume:
                iteration = self.args.save_iter
                init_model = torch.load(self.directory + "_iter = {iter}.pt".format(iter = iteration), map_location=torch.device(self.args.device))
                self.inner_lr = torch.load(self.directory + "_lr_iter = {iter}.pt".format(iter = iteration), map_location=torch.device(self.args.device))
                self.model.load_state_dict(init_model)


            while iteration < num_iterations + 1:
                tasks = self.tasks(mode='train', n_way=self.args.n_way, k_shot=self.args.k_shot, k_query=self.args.k_query, batchsz=10000, imsize=self.args.imsize, data_path=self.args.data_path)
                dataloader = DataLoader(tasks, self.meta_batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=False)

                for step, batch in enumerate(dataloader):
                    self.scheduler.step()

                    training_inputs = batch[0].to(self.args.device)
                    training_outputs = batch[1].to(self.args.device)
                    val_inputs = batch[2].to(self.args.device)
                    val_outputs = batch[3].to(self.args.device)

                    if training_inputs.shape[0] != self.meta_batch_size:
                        continue

                    self.meta_optimiser.zero_grad()

                    meta_loss = 0
                    losses = []

                    task_specific_parameter_list = [[w.clone() for w in self.weights] for _ in range(self.meta_batch_size)]

                    # inner-loop
                    for _ in range(self.step):
                        for i in range(self.meta_batch_size):
                            task_specific_parameter = task_specific_parameter_list[i]
                            # Compute training loss of current task
                            training_loss = self.criterion(self.model.parameterised(training_inputs[i], task_specific_parameter), training_outputs[i])

                            if self.penalty_type == 0:
                                pass
                            elif self.penalty_type == 1:
                                for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter):
                                    training_loss = training_loss + self.penalty_lr * self.penalty_criterion(w1 + w2 / self.num_tasks_mtrain, w3 / self.num_tasks_mtrain)
                            elif self.penalty_type == 2:
                                penalty_numerator = 0
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    penalty_numerator += self.penalty_criterion(w1, w2)
                                penalty_denominator = self.denominator + penalty_numerator
                                training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                            elif self.penalty_type == 3:
                                penalty_numerator = 0
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    penalty_numerator += self.penalty_criterion(w1, w2) ** 2
                                penalty_denominator = self.denominator + penalty_numerator
                                training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                            else:
                                raise RuntimeError('Invalid penalty-type index.')

                            # Computing task-specific parameter
                            grad = torch.autograd.grad(training_loss, task_specific_parameter, create_graph=True, retain_graph=True)
                            task_specific_parameter = [w - torch.mul(a, g) for w, a, g in zip(task_specific_parameter, self.inner_lr, grad)]

                            task_specific_parameter_list[i] = [w.clone() for w in task_specific_parameter]

                            # Update penalty parameters
                            if self.penalty_type == 1:
                                self.avg = [w1 + w2.detach().clone() / self.num_tasks_mtrain - w3.detach().clone() / self.num_tasks_mtrain for w1, w2, w3 in zip(self.avg, self.weights, task_specific_parameter)]

                            if self.penalty_type == 2:
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone()

                            if self.penalty_type == 3:
                                for w1, w2 in zip(self.weights, task_specific_parameter):
                                    self.denominator = self.denominator + self.penalty_criterion(w1, w2).detach().clone() ** 2

                    for i in range(self.meta_batch_size):
                        # Compute validation loss of current task
                        val_loss = self.criterion(self.model.parameterised(val_inputs[i], task_specific_parameter_list[i]), val_outputs[i])

                        losses.append(val_loss)

                    # Compute meta-loss
                    for i in range(self.meta_batch_size):
                        meta_loss += losses[i]
                    meta_loss = meta_loss / self.meta_batch_size

                    # Compute meta-gradient
                    meta_grads = torch.autograd.grad(meta_loss, self.weights + self.inner_lr)
                    with torch.no_grad():
                        for w, g in zip(self.weights + self.inner_lr, meta_grads):
                            w.grad = g.clamp(-1, 1) #w.grad = g
                    self.inner_lr = [torch.tensor(w - self.meta_lr * w.grad, requires_grad=True) for w in self.inner_lr]
                    # self.inner_lr = [torch.tensor(torch.clamp(w - self.meta_lr * w.grad, min=0, max=0.05), requires_grad=True) for w in self.inner_lr]

                    self.meta_optimiser.step()

                    # Reset penalty parameters
                    if self.penalty_type == 1:
                        self.avg = [torch.zeros(w.size(), requires_grad=False).to(self.args.device) for w in self.weights]

                    if self.penalty_type == 2 or self.penalty_type == 3:
                        self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

                    # Summaries
                    epoch_loss += meta_loss
                    if iteration % self.print_interval == 0:
                        print("{iter}/{total_iter}. loss: {loss}".format(iter=iteration, total_iter=num_iterations, loss=epoch_loss / self.plot_every / self.meta_batch_size))

                    if iteration % self.plot_every == 0:
                        self.meta_losses.append(epoch_loss / self.plot_every)
                        epoch_loss = 0

                    if iteration % self.save_interval == 0:
                        torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))
                        torch.save(self.inner_lr, self.directory + "_lr_iter = {iter}.pt".format(iter=iteration))

                    iteration += 1

                    if iteration > num_iterations + 1:
                        break


class CAVIA():
    def __init__(self, args, model, tasks, num_tasks_mtrain):
        self.args = args
        # Model
        self.model = model
        self.penalty_type = args.penalty_type

        # Model hyperparameter
        self.meta_lr = args.meta_lr
        self.inner_lr = args.inner_lr
        self.meta_batch_size = args.meta_batch_size
        self.update_batch_size = args.update_batch_size
        self.step = args.step

        # Datasource
        self.datasource = args.datasource
        self.tasks = tasks
        self.num_tasks_mtrain = num_tasks_mtrain

        # Optimizer
        self.weights = list(self.model.parameters())
        self.meta_optimiser = torch.optim.Adam(self.weights, lr=self.meta_lr)

        if self.datasource == 1:
            self.criterion = nn.MSELoss()
        elif self.datasource == 2:
            self.criterion = F.cross_entropy
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimiser, 5000, args.lr_meta_decay)

        # Log
        if self.datasource == 1:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta}, {inner})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size,
                task=self.update_batch_size, meta=self.meta_lr, inner=self.inner_lr)
        elif self.datasource == 2:
            self.directory = self.model.directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta}, {inner})".format(
                model=args.model_type, penalty=self.penalty_type, batch=self.meta_batch_size, n_way=args.n_way,
                k_shot=args.k_shot, k_query=args.k_query, meta=self.meta_lr, inner=self.inner_lr)

        self.plot_every = args.plot_log
        self.print_interval = args.print_log
        self.save_interval = args.save_log
        self.meta_losses = []

        # Penalty hyperparameter
        if self.penalty_type == 1:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight1 * (self.meta_batch_size * self.meta_batch_size / self.inner_lr / self.inner_lr / self.num_tasks_mtrain / self.num_tasks_mtrain)
            if self.datasource == 1:
                self.avg = torch.zeros(self.model.num_context_params, requires_grad=False)
            elif self.datasource == 2:
                self.avg = torch.zeros(self.model.num_context_params, requires_grad=False).to(self.args.device)

        if self.penalty_type == 2 or self.penalty_type == 3:
            self.penalty_criterion = nn.MSELoss(reduction='sum')
            self.penalty_lr = args.weight2 * self.meta_batch_size
            self.default = args.weight3
            if self.datasource == 1:
                self.init_params = torch.zeros(self.model.num_context_params, requires_grad=False)
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)
            elif self.datasource == 2:
                self.init_params = torch.zeros(self.model.num_context_params, requires_grad=False).to(self.args.device)
                self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

    def run_experiment(self, num_iterations):
        epoch_loss = 0
        self.model.train()

        if self.datasource == 1:
            for iteration in range(1, num_iterations + 1):
                self.meta_optimiser.zero_grad()

                meta_loss = 0
                losses, index = [], []
                gradient_list = [0 for _ in range(len(self.weights))]

                # inner-loop
                for _ in range(self.meta_batch_size):
                    self.model.reset_context_params()

                    idx, task = self.tasks.sample_task(train=True)
                    training_inputs, training_outputs = task.sample_data(K=self.update_batch_size)

                    # Compute training loss of current task
                    training_loss = self.criterion(self.model.parameterised(training_inputs, self.weights), training_outputs)

                    if self.penalty_type == 0:
                        pass
                    elif self.penalty_type == 1:
                        training_loss = training_loss + self.penalty_lr * self.penalty_criterion(self.avg, self.model.context_params / self.num_tasks_mtrain)
                    elif self.penalty_type == 2:
                        penalty_numerator = 0
                        penalty_numerator += self.penalty_criterion(self.model.context_params, self.init_params)
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    elif self.penalty_type == 3:
                        penalty_numerator = 0
                        penalty_numerator += self.penalty_criterion(self.model.context_params, self.init_params) ** 2
                        penalty_denominator = self.denominator + penalty_numerator
                        training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                    else:
                        raise RuntimeError('Invalid penalty-type index.')

                    # Computing task-specific parameter
                    grad = torch.autograd.grad(training_loss, self.model.context_params, create_graph=True)[0]
                    self.model.context_params = self.model.context_params - self.inner_lr * grad

                    # Update penalty parameters
                    if self.penalty_type == 1:
                        self.avg = self.avg - self.model.context_params.detach().clone() / self.num_tasks_mtrain

                    if self.penalty_type == 2:
                        self.denominator = self.denominator + self.penalty_criterion(self.model.context_params, self.init_params).detach().clone()

                    if self.penalty_type == 3:
                        self.denominator = self.denominator + self.penalty_criterion(self.model.context_params, self.init_params).detach().clone() ** 2

                    # Compute validation loss of current task
                    val_inputs, val_outputs = task.sample_data(K=self.update_batch_size)
                    val_loss = self.criterion(self.model.parameterised(val_inputs, self.weights), val_outputs)

                    task_grad = torch.autograd.grad(val_loss, self.weights)
                    for i in range(len(task_grad)):
                        gradient_list[i] += task_grad[i].detach().clamp_(-10, 10)
                    losses.append(val_loss.item())
                    index.append(idx)

                # Compute meta-loss
                for i in range(self.meta_batch_size):
                    meta_loss += losses[i]

                # Compute meta-gradient
                with torch.no_grad():
                    for w, g in zip(self.weights, gradient_list):
                        w.grad = g

                self.meta_optimiser.step()

                # Reset penalty parameters
                if self.penalty_type == 1:
                    self.avg = torch.zeros(self.model.num_context_params, requires_grad=False)

                if self.penalty_type == 2 or self.penalty_type == 3:
                    self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False)

                # Summaries
                epoch_loss += meta_loss
                if iteration % self.print_interval == 0:
                    print("{iter}/{total_iter}. loss: {loss}".format(iter=iteration, total_iter=num_iterations, loss=epoch_loss / self.plot_every / self.meta_batch_size))

                if iteration % self.plot_every == 0:
                    self.meta_losses.append(epoch_loss / self.plot_every)
                    epoch_loss = 0

                if iteration % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))

        elif self.datasource == 2:
            iteration = 1

            if self.args.resume:
                iteration = self.args.save_iter
                init_model = torch.load(self.directory + "_iter = {iter}.pt".format(iter = iteration), map_location=torch.device(self.args.device))
                self.model.load_state_dict(init_model)

            while iteration < num_iterations + 1:
                tasks = self.tasks(mode='train', n_way=self.args.n_way, k_shot=self.args.k_shot, k_query=self.args.k_query, batchsz=10000, imsize=self.args.imsize, data_path=self.args.data_path)
                dataloader = DataLoader(tasks, self.meta_batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=False)

                for step, batch in enumerate(dataloader):
                    self.scheduler.step()

                    training_inputs = batch[0].to(self.args.device)
                    training_outputs = batch[1].to(self.args.device)
                    val_inputs = batch[2].to(self.args.device)
                    val_outputs = batch[3].to(self.args.device)

                    if training_inputs.shape[0] != self.meta_batch_size:
                        continue

                    self.meta_optimiser.zero_grad()

                    meta_loss = 0
                    losses = []
                    gradient_list = [0 for _ in range(len(self.weights))]

                    self.model.reset_context_params()
                    context_params_list = [self.model.context_params.clone() for _ in range(self.meta_batch_size)]

                    # inner-loop
                    for _ in range(self.step):
                        for i in range(self.meta_batch_size):
                            self.model.context_params = context_params_list[i]

                            # Compute training loss of current task
                            training_loss = self.criterion(self.model(training_inputs[i]), training_outputs[i])

                            if self.penalty_type == 0:
                                pass
                            elif self.penalty_type == 1:
                                training_loss = training_loss + self.penalty_lr * self.penalty_criterion(self.avg, self.model.context_params / self.num_tasks_mtrain)
                            elif self.penalty_type == 2:
                                penalty_numerator = 0
                                penalty_numerator += self.penalty_criterion(self.model.context_params, self.init_params)
                                penalty_denominator = self.denominator + penalty_numerator
                                training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                            elif self.penalty_type == 3:
                                penalty_numerator = 0
                                penalty_numerator += self.penalty_criterion(self.model.context_params, self.init_params) ** 2
                                penalty_denominator = self.denominator + penalty_numerator
                                training_loss = training_loss + self.penalty_lr * penalty_numerator / penalty_denominator
                            else:
                                raise RuntimeError('Invalid penalty-type index.')

                            # Computing task-specific parameter
                            grad = torch.autograd.grad(training_loss, self.model.context_params, create_graph=True)[0]
                            self.model.context_params = self.model.context_params - self.inner_lr * grad

                            context_params_list[i] = self.model.context_params.clone()

                            # Update penalty parameters
                            if self.penalty_type == 1:
                                self.avg = self.avg - self.model.context_params.detach().clone() / self.num_tasks_mtrain

                            if self.penalty_type == 2:
                                self.denominator = self.denominator + self.penalty_criterion(self.model.context_params, self.init_params).detach().clone()

                            if self.penalty_type == 3:
                                self.denominator = self.denominator + self.penalty_criterion(self.model.context_params, self.init_params).detach().clone() ** 2


                    for i in range(self.meta_batch_size):
                        self.model.context_params = context_params_list[i]

                        # Compute validation loss of current task
                        val_loss = self.criterion(self.model(val_inputs[i]), val_outputs[i])

                        task_grad = torch.autograd.grad(val_loss, self.weights)
                        for j in range(len(task_grad)):
                            gradient_list[j] += task_grad[j].detach().clamp(-10, 10)
                        losses.append(val_loss.item())

                    # Compute meta-loss
                    for i in range(self.meta_batch_size):
                        meta_loss += losses[i]

                    # Compute meta-gradient
                    with torch.no_grad():
                        for w, g in zip(self.weights, gradient_list):
                            w.grad = g

                    self.meta_optimiser.step()

                    # Reset penalty parameters
                    if self.penalty_type == 1:
                        self.avg = torch.zeros(self.model.num_context_params, requires_grad=False).to(self.args.device)

                    if self.penalty_type == 2 or self.penalty_type == 3:
                        self.denominator = torch.tensor(self.default, dtype=torch.float32, requires_grad=False).to(self.args.device)

                    # Summaries
                    epoch_loss += meta_loss
                    if iteration % self.print_interval == 0:
                        print("{iter}/{total_iter}. loss: {loss}".format(iter=iteration, total_iter=num_iterations, loss=epoch_loss / self.plot_every / self.meta_batch_size))

                    if iteration % self.plot_every == 0:
                        self.meta_losses.append(epoch_loss / self.plot_every)
                        epoch_loss = 0

                    if iteration % self.save_interval == 0:
                        torch.save(self.model.state_dict(), self.directory + "_iter = {iter}.pt".format(iter=iteration))

                    iteration += 1

                    if iteration > num_iterations + 1:
                        break