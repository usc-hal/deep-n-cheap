import argparse
import os
from utile import printf, dict_str
import torch
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--network", type=str, default='nlp', help="Choose from 'cnn' or 'mlp' or 'nlp'")
parser.add_argument("--dataset", type=str, default='yelp', help="Either something like 'mnist', or something like 'mnist.npz'")
parser.add_argument("--input_size", type=int, nargs='+', default=[30002], help="Dimensions of 1 input data sample")
parser.add_argument("--output_size", type=int, default=5, help="Sumber of classes")
parser.add_argument("--data_folder", type=str, default='/home/babakniy/', help="Path to folder where dataset is stored")
parser.add_argument("--augment", type=bool, default=True, help="<CNNs only> Whether to apply basic augmentation")
parser.add_argument("--verbose", type=bool, default=True, help="print messages after every epoch")
parser.add_argument("--val_split", type=float, default=0.08, help="Fraction of complete training data to use for validation")

parser.add_argument("--wc", type=float, default=0, help="Complexity penalty coefficient")
parser.add_argument("--penalize", type=str, default='t_epoch', help="<MLPs only> Choose from 't_epoch' for training time per epoch, or 'numparams' for number of trainable parameters (CNNs only support t_epoch)")
parser.add_argument("--tbar_epoch", type=float, default='720', help="Baseline time to normalize t_epoch. Platform dependent.")
parser.add_argument("--numepochs", type=int, default=20, help="How many epochs to run each model for")
parser.add_argument("--val_patience", type=int, default=4, help="Terminate training if val accuracy doesn't increase for this many epochs. The default is basically infinity.")

parser.add_argument("--bo_prior_states", type=int, default=15, help="# prior states for Bayesian optimization")
parser.add_argument("--bo_steps", type=int, default=15, help="# optimization steps for Bayesian optimization")
parser.add_argument("--bo_explore", type=int, default=1000, help="# states each step of acquisition function explores in Bayesian optimization")
# parser.add_argument("--grid_search_order", type=str, nargs=5, default=['ac', 'ds', 'bn', 'do', 'sc'], help="<CNNs only> Any permutation of downsampling 'ds', batch normalization 'bn', dropout 'do' and shortcut connections 'sc'")
parser.add_argument("--grid_search_order", type=str, nargs=5, default=['do', 'sc'], help="<CNNs only> Any permutation of downsampling 'ds', batch normalization 'bn', dropout 'do' and shortcut connections 'sc'")

parser.add_argument("--num_conv_layers", type=int, nargs=2, default=[2, 20], metavar=('lower_limit', 'upper_limit'), help="<CNNs only> Limits for # conv layers")
parser.add_argument("--channels_first", type=int, nargs=2, default=[50, 300], metavar=('lower_limit', 'upper_limit'), help="<CNNs only> Limits for # channels in 1st conv layer")
parser.add_argument("--embedding_dim", type=int, nargs=2, default=[70, 300], metavar=('lower_limit', 'upper_limit'), help="<NLP only> Limits for # embedding size")
parser.add_argument("--channels_upper", type=int, default=350, help="<CNNs only> Upper limit for # channels in any conv layer")
parser.add_argument("--num_hidden_layers", type=int, nargs=2, default=[0, 2], metavar=('lower_limit', 'upper_limit'), help="<MLPs only> Limits for # hidden layers")
parser.add_argument("--hidden_nodes", type=int, nargs=2, default=[20, 400], metavar=('lower_limit', 'upper_limit'), help="<MLPs only> Limits for # hidden nodes")
parser.add_argument("--lr", type=float, nargs=2, default=[-5, -1], metavar=('lower_log_limit', 'upper_log_limit'), help="Log limits for learning rate")
parser.add_argument("--weight_decay", type=float, nargs=2, default=[-5, -3], metavar=('lower_log_limit', 'upper_log_limit'), help="Log limits for weight decay (lowest order of magnitude converted to 0 weight decay)")
parser.add_argument("--batch_size", type=int, nargs=2, default=[100, 100], metavar=('lower_limit', 'upper_limit'), help="Limits for batch size")

parser.add_argument("--bn_fracs", type=float, nargs='+', default=[0], help="<CNNs only> What fractions of batch norm layers to explore")
parser.add_argument("--do_fracs", type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1], help="<CNNs only> What fractions of dropout layers to explore")
parser.add_argument("--input_drop_probs", type=float, nargs='+', default=[0.1, 0.2], help="<CNNs only> What drop probabilities to explore in input layer")
parser.add_argument("--drop_probs_cnn", type=float, nargs='+', default=[0.15, 0.45], help="<CNNs only> What drop probabilities to explore in non-input layers")
parser.add_argument("--drop_probs_mlp", type=float, nargs='+', default=[0, 0.1, 0.2, 0.3, 0.4, 0.5], help="<MLPs only> What drop probabilities to explore in hidden layers")
parser.add_argument("--num_best", type=int, default=1, help="# best models to return from final stage of search")
parser.add_argument("--prior_time", type=float, default=0.0, help="When resuming a run, enter time elapsed so far (useful for getting accurate time estimate)")

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--is_dpcnn", type=str, default='1')
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--note", type=str, default='dpcnn')

# DL framework
parser.add_argument("--dl_framework", type=str, default='torch', help="Choose from 'torch' or 'tf.keras'")
# task/problem type. Currently support classification and regression.
# TODO: detection, tracking, segmentation, etc.
parser.add_argument("--problem_type", type=str, default='classification', help="Choose from 'classification' or 'regression'")

args = parser.parse_args()

if args.dl_framework == 'torch':
    os.environ['DNC_DL_FRAMEWORK'] = 'torch'
    os.environ['is_DPCNN'] = args.is_dpcnn
elif args.dl_framework == 'tf.keras':
    os.environ['DNC_DL_FRAMEWORK'] = 'tf.keras'
else:
    raise Exception("framework not support!!!")


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

printf("Feb, 19")
printf(dict_str(args.__dict__))
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.network == "nlp":
    from data.prepare_nlp_data import get_saved_data
    dataset = args.dataset + f'.{args.output_size}'
    data = get_saved_data(dataset, args.batch_size[0], args.data_folder, args.val_split, args.seed)
    # from torch.utils.data import TensorDataset, DataLoader
    # from torch.utils.data import RandomSampler
    # tx = torch.randint(1, 150, (64, 100))
    # ty = torch.randint(0, 2, (64,))
    # td = TensorDataset(tx, ty)
    # ts = RandomSampler(td)
    # train_dataloader = DataLoader(td, sampler=ts, batch_size=args.batch_size[0])
    # data = {
    #     'train': train_dataloader,
    #     'val': train_dataloader,
    #     'test': train_dataloader,
    #     'type': 'loader'
    # }

elif args.dataset[-4:] == '.npz':
    # use .npz files
    from data.data import get_data_npz
    dataset = args.dataset[:-4]
    data = get_data_npz(data_folder=args.data_folder, dataset=args.dataset, val_split=args.val_split, problem_type=args.problem_type)
else:
    # use framework built-in datasets
    from data.data import get_data
    dataset = args.dataset
    data = get_data(data_folder=args.data_folder, dataset=args.dataset, val_split=args.val_split, augment=args.augment)

dataset_code = 'M' if dataset == 'mnist' else 'F' if dataset == 'fmnist' else 'R' if dataset == 'rcv1_2000' else 'Y' if dataset == 'yelp' else 'S' if dataset == 'SST' else 'XXX'

if args.network == 'cnn':
    from model_search.CNN import run_model_search_cnn
    run_model_search_cnn(data=data, dataset_code=dataset_code,
                         input_size=args.input_size, output_size=args.output_size, problem_type=args.problem_type, verbose=args.verbose,
                         wc=args.wc, tbar_epoch=args.tbar_epoch, numepochs=args.numepochs, val_patience=args.val_patience,
                         bo_prior_states=args.bo_prior_states, bo_steps=args.bo_steps, bo_explore=args.bo_explore, grid_search_order=args.grid_search_order,
                         num_conv_layers=args.num_conv_layers, channels_first=args.channels_first, channels_upper=args.channels_upper, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                         bn_fracs=args.bn_fracs, do_fracs=args.do_fracs, input_drop_probs=args.input_drop_probs, drop_probs=args.drop_probs_cnn,
                         num_best=args.num_best, prior_time=args.prior_time)

elif args.network == 'mlp':
    from model_search.mlp import run_model_search_mlp
    run_model_search_mlp(data=data, dataset_code=dataset_code,
                         input_size=args.input_size, output_size=args.output_size, problem_type=args.problem_type, verbose=args.verbose,
                         wc=args.wc, penalize=args.penalize, tbar_epoch=args.tbar_epoch, numepochs=args.numepochs, val_patience=args.val_patience,
                         bo_prior_states=args.bo_prior_states, bo_steps=args.bo_steps, bo_explore=args.bo_explore,
                         num_hidden_layers=args.num_hidden_layers, hidden_nodes=args.hidden_nodes, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                         drop_probs=args.drop_probs_mlp,
                         num_best=args.num_best, prior_time=args.prior_time)

elif args.network == 'nlp':
    from model_search.nlp import run_model_search_nlp
    run_model_search_nlp(data=data, dataset_code=dataset_code,
                         input_size=args.input_size, output_size=args.output_size, problem_type=args.problem_type, verbose=args.verbose,
                         wc=args.wc, tbar_epoch=args.tbar_epoch, numepochs=args.numepochs, val_patience=args.val_patience,
                         bo_prior_states=args.bo_prior_states, bo_steps=args.bo_steps, bo_explore=args.bo_explore, grid_search_order=args.grid_search_order,
                         num_conv_layers=args.num_conv_layers, channels_first=args.channels_first, channels_upper=args.channels_upper, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                         bn_fracs=args.bn_fracs, do_fracs=args.do_fracs, input_drop_probs=args.input_drop_probs, drop_probs=args.drop_probs_cnn,
                         num_best=args.num_best, prior_time=args.prior_time, embedding_dim=args.embedding_dim, is_dpcnn=args.is_dpcnn == '1')
