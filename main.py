import arguments
import utils
from datasource import Sine_Task_Distribution, MiniImageNet
from model import MAMLModel, MAMLConvModel, CAVIAModel, CAVIAConvModel, MAML, TRMAML, Reptile, MetaSGD, CAVIA
from eval import evaluation

import torch

if __name__ == '__main__':
    args = arguments.parse_args()

    if args.datasource == 1:
        tasks = Sine_Task_Distribution()
        num_tasks_mtrain, num_tasks_mtest = tasks.num_tasks()
        if args.train:
            utils.set_seed(args.seed)
            if args.model_type == 1:
                Model = MAML(args=args,
                             model=MAMLModel(n_input=args.n_input, n_output=args.n_output, n_hidden=args.n_hidden),
                             tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)

            elif args.model_type == 2:
                Model = TRMAML(args=args,
                             model=MAMLModel(n_input=args.n_input, n_output=args.n_output, n_hidden=args.n_hidden),
                             tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)

            elif args.model_type == 3:
                Model = Reptile(args=args,
                               model=MAMLModel(n_input=args.n_input, n_output=args.n_output, n_hidden=args.n_hidden),
                               tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)

            elif args.model_type == 4:
                Model = MetaSGD(args=args,
                               model=MAMLModel(n_input=args.n_input, n_output=args.n_output, n_hidden=args.n_hidden),
                               tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)

            elif args.model_type == 5:
                Model = CAVIA(args=args,
                              model=CAVIAModel(n_input=args.n_input, n_output=args.n_output, n_hidden=args.n_hidden,
                                               num_context_params=args.num_context_params),
                              tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)

            else:
                raise RuntimeError('Invalid model-type index.')
            # Run Model
            Model.run_experiment(num_iterations=args.num_iterations)

            # Save Model
            if args.model_type == 4:
                torch.save(Model.inner_lr, Model.directory + "_lr.pt")

            torch.save(Model.model.state_dict(), Model.directory + ".pt")
        else:
            directory = "Sinusoid"
            if args.model_type == 1:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta}, {inner})".format(
            model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, task=args.update_batch_size,
            meta=args.meta_lr, inner=args.inner_lr)

            elif args.model_type == 2:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta}, {inner}, {prob})".format(
            model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, task=args.update_batch_size,
            meta=args.meta_lr, inner=args.inner_lr, prob=args.p_lr)

            elif args.model_type == 3:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta}, {inner})".format(
            model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, task=args.update_batch_size,
            meta=args.epsilon, inner=args.inner_lr)

            elif args.model_type == 4:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta})".format(
            model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, task=args.update_batch_size,
            meta=args.meta_lr)

            elif args.model_type == 5:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {task})_lr=({meta}, {inner})".format(
            model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, task=args.update_batch_size,
            meta=args.meta_lr, inner=args.inner_lr)

            else:
                raise RuntimeError('Invalid model-type index.')

            avg_losses, worst_losses, std_devs = evaluation(args=args, tasks=tasks, dir=directory, num_tasks_mtest=num_tasks_mtest)

            f = open(directory + ".csv", "w")
            f.write(str(avg_losses) + ',' + str(worst_losses) + ',' + str(std_devs) + '\n')
            f.close()

    elif args.datasource == 2:
        tasks = MiniImageNet
        num_tasks_mtrain = 64 ### 64 C N
        num_tasks_mtest = 24 ### 36
        if args.train:
            utils.set_seed(args.seed)
            if args.model_type == 1:
                Model = MAML(args=args, model=MAMLConvModel(num_classes=args.n_way, num_filters=args.num_filters,
                                                             max_pool=not args.no_max_pool,
                                                             imsize=args.imsize, initialisation=args.nn_init,
                                                             device=args.device), tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)
            elif args.model_type == 2:
                Model = TRMAML(args=args, model=MAMLConvModel(num_classes=args.n_way, num_filters=args.num_filters,
                                                            max_pool=not args.no_max_pool,
                                                            imsize=args.imsize, initialisation=args.nn_init,
                                                            device=args.device), tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)
            elif args.model_type == 3:
                Model = Reptile(args=args, model=MAMLConvModel(num_classes=args.n_way, num_filters=args.num_filters,
                                                            max_pool=not args.no_max_pool,
                                                            imsize=args.imsize, initialisation=args.nn_init,
                                                            device=args.device), tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)
            elif args.model_type == 4:
                Model = MetaSGD(args=args, model=MAMLConvModel(num_classes=args.n_way, num_filters=args.num_filters,
                                                            max_pool=not args.no_max_pool,
                                                            imsize=args.imsize, initialisation=args.nn_init,
                                                            device=args.device), tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)

            elif args.model_type == 5:
                Model = CAVIA(args=args, model=CAVIAConvModel(num_classes=args.n_way, num_filters=args.num_filters,
                                                             max_pool=not args.no_max_pool,
                                                             num_context_params=args.num_context_params,
                                                             context_in=args.context_in,
                                                             num_film_hidden_layers=args.num_film_hidden_layers,
                                                             imsize=args.imsize, initialisation=args.nn_init,
                                                             device=args.device), tasks=tasks, num_tasks_mtrain=num_tasks_mtrain)
            else:
                raise RuntimeError('Invalid model-type index.')
            # Run Model
            Model.run_experiment(num_iterations=args.num_iterations)

            # Save Model
            if args.model_type == 4:
                torch.save(Model.inner_lr, Model.directory + "_lr.pt")

            torch.save(Model.model.state_dict(), Model.directory + ".pt")
        else:
            directory = "MiniImageNet"
            if args.model_type == 1:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta}, {inner})".format(
                    model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, n_way=args.n_way,
                    k_shot=args.k_shot, k_query=args.k_query, meta=args.meta_lr, inner=args.inner_lr)
            elif args.model_type == 2:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta}, {inner}, {prob})".format(
                    model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, n_way=args.n_way,
                    k_shot=args.k_shot, k_query=args.k_query, meta=args.meta_lr, inner=args.inner_lr, prob=args.p_lr)
            elif args.model_type == 3:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta}, {inner})".format(
                    model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, n_way=args.n_way,
                    k_shot=args.k_shot, k_query=args.k_query, meta=args.epsilon, inner=args.inner_lr)
            elif args.model_type == 4:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta})".format(
                    model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, n_way=args.n_way,
                    k_shot=args.k_shot, k_query=args.k_query, meta=args.meta_lr)
            elif args.model_type == 5:
                directory = directory + "/model=({model}, {penalty})_batch_size=({batch}, {n_way}, {k_shot}, {k_query})_lr=({meta}, {inner})".format(
                model=args.model_type, penalty=args.penalty_type, batch=args.meta_batch_size, n_way=args.n_way,
                k_shot=args.k_shot, k_query=args.k_query, meta=args.meta_lr, inner=args.inner_lr)
            else:
                raise RuntimeError('Invalid model-type index.')

            avg_losses, _, _ = evaluation(args=args, tasks=tasks, dir=directory)

            f = open(directory + ".csv", "w")
            f.write(str(avg_losses))
            f.close()
                                
    else:
        raise RuntimeError('Invalid datasource index.')
