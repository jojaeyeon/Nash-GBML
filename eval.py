from model import MAMLModel, CAVIAModel, CAVIAConvModel, MAMLConvModel
from torch.utils.data import DataLoader

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

def loss_on_task(args, init_model, tasks, inner_lr):
    if args.model_type == 5:
        model = CAVIAModel(n_input=args.n_input, n_output=args.n_output, n_hidden=args.n_hidden,
                                               num_context_params=args.num_context_params)
    else:
        model = MAMLModel(n_input=args.n_input, n_output=args.n_output, n_hidden=args.n_hidden)

    model.load_state_dict(init_model)
    model.eval()

    criterion = nn.MSELoss()

    # train model on a random task
    index, task = tasks.sample_task(train=False)
    weights = list(model.parameters())

    if args.model_type == 5:
        model.reset_context_params()
        inputs, outputs = task.sample_data(K=args.update_batch_size)
        loss = criterion(model.parameterised(inputs, weights), outputs)
        grad = torch.autograd.grad(loss, model.context_params)[0]
        model.context_params = model.context_params - inner_lr * grad
    else:
        inputs, outputs = task.sample_data(K=args.update_batch_size)
        loss = criterion(model.parameterised(inputs, weights), outputs)
        grad = torch.autograd.grad(loss, weights)
        if args.model_type == 4:
            weights = [w - torch.mul(a, g) for w, a, g in zip(weights, inner_lr, grad)]
        else:
            weights = [w - inner_lr * g for w, g in zip(weights, grad)]

    val_inputs, val_outputs = task.sample_data(K=args.test_samples)
    val_loss = criterion(model.parameterised(val_inputs, weights), val_outputs)

    return index, val_loss.item()


def evaluation(args, tasks, dir, num_tasks_mtest=None):
    init_model = torch.load(dir + ".pt", map_location=torch.device(args.device))
    if args.model_type == 4:
        if args.datasource == 1:
            inner_lr = torch.load(dir + "_lr.pt")
        else:
            inner_lr = torch.load(dir + "_lr.pt", map_location=torch.device(args.device))
    else:
        inner_lr = args.inner_lr

    if args.datasource == 1:
        test_loss = np.zeros(num_tasks_mtest)
        idx_count = np.ones(num_tasks_mtest) * 0.5

        for i in range(args.n_samples):
            index, loss = loss_on_task(args, init_model, tasks, inner_lr)
            idx_count[index] = math.floor(idx_count[index] + 1)
            test_loss[index] += loss

        test_loss = np.divide(test_loss, idx_count)
        avg_loss = np.mean(test_loss)
        worst_loss = np.max(test_loss)
        std_devs = np.std(test_loss)

        return avg_loss, worst_loss, std_devs

    elif args.datasource == 2:
        if args.model_type == 5:
            model = CAVIAConvModel(num_classes=args.n_way, num_filters=args.num_filters,
                                                             max_pool=not args.no_max_pool,
                                                             num_context_params=args.num_context_params,
                                                             context_in=args.context_in,
                                                             num_film_hidden_layers=args.num_film_hidden_layers,
                                                             imsize=args.imsize, initialisation=args.nn_init,
                                                             device=args.device)
        else:
            model = MAMLConvModel(num_classes=args.n_way, num_filters=args.num_filters,
                                                             max_pool=not args.no_max_pool,
                                                             imsize=args.imsize, initialisation=args.nn_init,
                                                             device=args.device)

        model.load_state_dict(init_model)
        model.eval()

        criterion = F.cross_entropy

        dataset = tasks(mode='test', n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query, batchsz=500, verbose=False, imsize=args.imsize, data_path=args.data_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

        test_loss = []

        for c, batch in enumerate(dataloader):
            support_x = batch[0].to(args.device)
            support_y = batch[1].to(args.device)
            query_x = batch[2].to(args.device)
            query_y = batch[3].to(args.device)

            for i in range(support_x.shape[0]):
                if args.model_type == 5:
                    model.reset_context_params()
                    for step in range(args.step):
                        loss = criterion(model(support_x[i]), support_y[i])
                        grad = torch.autograd.grad(loss, model.context_params, create_graph=True)[0]
                        model.context_params = model.context_params - inner_lr * grad
                    test_loss.append(torch.argmax(F.softmax(model(query_x[i]), dim=1), 1).eq(query_y[i]).sum().item() / len(query_y[i]))
                else:
                    weights = list(model.parameters())
                    for step in range(args.step):
                        loss = criterion(model.parameterised(support_x[i], weights), support_y[i])
                        grad = torch.autograd.grad(loss, weights, create_graph=True)
                        if args.model_type == 4:
                            weights = [w - torch.mul(a, g) for w, a, g in zip(weights, inner_lr, grad)]
                        else:
                            weights = [w - inner_lr * g for w, g in zip(weights, grad)]
                    test_loss.append(torch.argmax(F.softmax(model.parameterised(query_x[i], weights), dim=1), 1).eq(query_y[i]).sum().item() / len(query_y[i]))

        avg_loss = np.mean(test_loss)
        worst_loss = np.max(test_loss)
        std_devs = np.std(test_loss)

        return avg_loss, worst_loss, std_devs
