import time
import numpy as np
import pickle

from model_search.model_search_helper import distancefunc_ramp, kernelfunc_se, activation, default_weight_decay
from model_search.model_search_helper import bayesopt, net_kws_defaults, run_kws_defaults, dropout_mlp
from .utile import printf


def run_model_search_mlp(data, dataset_code,
                         input_size, output_size, problem_type, verbose,
                         wc, penalize, tbar_epoch, numepochs, val_patience,
                         bo_prior_states, bo_steps, bo_explore,
                         num_hidden_layers, hidden_nodes, lr, weight_decay, batch_size,
                         drop_probs,
                         num_best, prior_time):

    # Data, etc ##
    run_network_kw = {
        'data': data,
        'input_size': input_size,
        'output_size': output_size,
        'problem_type': problem_type,
        'verbose': verbose
    }

    # Covmat params ##
    distancefunc = distancefunc_ramp
    kernelfunc = kernelfunc_se

    # BO params ##
    out_channels_bo = {'steps': bo_steps, 'initial_states_size': bo_prior_states, 'new_states_size': bo_explore}
    training_hyps_bo = {'steps': bo_steps, 'initial_states_size': bo_prior_states, 'new_states_size': bo_explore, 'num_best': num_best}

    # State params ##
    get_states_limits = {'num_hidden_layers_mlp': num_hidden_layers, 'hidden_nodes_mlp': hidden_nodes,
                         'lr': lr, 'weight_decay': weight_decay, 'batch_size': batch_size}

    if penalize == 't_epoch':
        penalize_bar = tbar_epoch
    else:  # numparams
        penalize_bar = get_states_limits['hidden_nodes_mlp'][1] * (run_network_kw['input_size'][0] + run_network_kw['output_size'] + (get_states_limits['num_hidden_layers_mlp'][1] - 1) * get_states_limits['hidden_nodes_mlp'][1])  # total weights
        penalize_bar += (get_states_limits['num_hidden_layers_mlp'][1] * get_states_limits['hidden_nodes_mlp'][1] + run_network_kw['output_size'])  # total biases

    start_time = time.time()
# =============================================================================
#     Step 1: Bayesian optimization for number of hidden layers and nodes
# =============================================================================
    printf('STARTING hidden_mlp')
    the_best_states, the_best_loss_stats = bayesopt(
        state_kw={
            'state_keys': ['hidden_mlp'],
            'limits': get_states_limits
        },
        loss_kw={
            'net_kw_const': {},
            'run_kw_const': {},
            'val_patience': val_patience,
            'numepochs': numepochs,
            'dataset_code': dataset_code,
            'run_network_kw': run_network_kw,
            'penalize': penalize,
            'wc': wc,
            'tbar_epoch' if penalize == 't_epoch' else 'numparams_bar': penalize_bar,
            'problem_type': problem_type
        },
        mu_val=None,
        covmat_kw={
            'distancefunc': distancefunc,
            'kernelfunc': kernelfunc,
            'limits': {
                'hidden_mlp': [(0, get_states_limits['num_hidden_layers_mlp'][1] * get_states_limits['hidden_nodes_mlp'][1]), get_states_limits['num_hidden_layers_mlp']]
            },
        },
        **out_channels_bo)
    printf('TOTAL SEARCH TIME = {0}\n\n'.format(time.time() - start_time + prior_time))


# =============================================================================
#     Step 2: Fine-tuning architecture using grid search
# =============================================================================

    # Initialization ##
    the_best_state = the_best_states[0]
    the_best_loss = the_best_loss_stats[0]['loss']
    if problem_type == 'classification':
        the_best_loss_val_acc = the_best_loss_stats[0]['best_val_acc']
    elif problem_type == 'regression':
        the_best_loss_val_loss = the_best_loss_stats[0]['best_val_loss']
    the_best_loss_penalize = the_best_loss_stats[0][penalize]
    num_hidden_layers = len(the_best_state['hidden_mlp'])

    # Dropout MLP ##
    printf('STARTING activation')
    best_state, best_loss, loss_stats = activation(
        numlayers=num_hidden_layers,
        loss_kw={
            'net_kw_const': the_best_state,
            'run_kw_const': {},
            'val_patience': val_patience,
            'numepochs': numepochs,
            'dataset_code': dataset_code,
            'run_network_kw': run_network_kw,
            'penalize': penalize,
            'wc': wc,
            'tbar_epoch' if penalize == 't_epoch' else 'numparams_bar': penalize_bar,
            'problem_type': problem_type
        })

    if best_loss < the_best_loss:
        the_best_state.update(best_state)
        the_best_loss = best_loss
        if problem_type == 'classification':
            the_best_loss_val_acc = loss_stats['best_val_acc']
        elif problem_type == 'regression':
            the_best_loss_val_loss = loss_stats['best_val_loss']
        the_best_loss_penalize = loss_stats[penalize]
    else:
        the_best_state.update({'act': 'relu'})
    if problem_type == 'classification':
        printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and {3} = {4}, TOTAL SEARCH TIME = {5}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, penalize, the_best_loss_penalize, time.time() - start_time + prior_time))
    elif problem_type == 'regression':
        printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_LOSS = {2} and {3} = {4}, TOTAL SEARCH TIME = {5}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_loss, penalize, the_best_loss_penalize, time.time() - start_time + prior_time))

    printf('STARTING dropout')
    best_state, best_loss, loss_stats = dropout_mlp(
        num_hidden_layers=num_hidden_layers,
        drop_probs=drop_probs,
        loss_kw={
            'net_kw_const': the_best_state,
            'run_kw_const': {},
            'val_patience': val_patience,
            'numepochs': numepochs,
            'dataset_code': dataset_code,
            'run_network_kw': run_network_kw,
            'penalize': penalize,
            'wc': wc,
            'tbar_epoch' if penalize == 't_epoch' else 'numparams_bar': penalize_bar,
            'problem_type': problem_type})
    if best_loss < the_best_loss:
        the_best_state.update(best_state)
        the_best_loss = best_loss
        if problem_type == 'classification':
            the_best_loss_val_acc = loss_stats['best_val_acc']
        elif problem_type == 'regression':
            the_best_loss_val_loss = loss_stats['best_val_loss']
        the_best_loss_penalize = loss_stats[penalize]
    else:
        the_best_state.update({'apply_dropouts_mlp': num_hidden_layers * net_kws_defaults['apply_dropouts_mlp'],
                               'dropout_probs_mlp': num_hidden_layers * net_kws_defaults['dropout_probs_mlp']})
    if problem_type == 'classification':
        printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and {3} = {4}, TOTAL SEARCH TIME = {5}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, penalize, the_best_loss_penalize, time.time() - start_time + prior_time))
    elif problem_type == 'regression':
        printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_LOSS = {2} and {3} = {4}, TOTAL SEARCH TIME = {5}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_loss, penalize, the_best_loss_penalize, time.time() - start_time + prior_time))


# =============================================================================
#     Step 3: Bayesian optimization for training hyperparameters
# =============================================================================
    printf('STARTING training hyperparameters')
    final_best_states, final_best_loss_stats = bayesopt(
        state_kw={
            'state_keys': ['lr', 'weight_decay', 'batch_size'],
            'limits': get_states_limits},
        loss_kw={
            'net_kw_const': the_best_state,
            'run_kw_const': {},
            'val_patience': val_patience,
            'numepochs': numepochs,
            'dataset_code': dataset_code,
            'run_network_kw': run_network_kw,
            'penalize': penalize,
            'wc': wc,
            'tbar_epoch' if penalize == 't_epoch' else 'numparams_bar': penalize_bar,
            'problem_type': problem_type},
        mu_val=None,
        covmat_kw={
            'distancefunc': distancefunc,
            'kernelfunc': kernelfunc,
            'limits': get_states_limits,  # those are also valid for training hyps
        },
        **training_hyps_bo)

    # Append architectures ##
    for i in range(len(final_best_states)):
        final_best_states[i] = {**the_best_state, **final_best_states[i]}

    # Calculate number of parameters and default weight decay, then append existing state to final ##
    final_best_states.append({**the_best_state,
                              **{'lr': run_kws_defaults['lr'],
                                 'weight_decay': default_weight_decay(dataset_code=dataset_code, input_size=run_network_kw['input_size'], output_size=run_network_kw['output_size'], net_kw=the_best_state),
                                 'batch_size': run_kws_defaults['batch_size']}
                              })
    if problem_type == 'classification':
        final_best_loss_stats.append({'loss': the_best_loss, 'best_val_acc': the_best_loss_val_acc, penalize: the_best_loss_penalize})
    elif problem_type == 'regression':
        final_best_loss_stats.append({'loss': the_best_loss, 'best_val_loss': the_best_loss_val_loss, penalize: the_best_loss_penalize})

# =============================================================================
#     Final stats and records
# =============================================================================
    final_losses = np.asarray([fbls['loss'] for fbls in final_best_loss_stats])
    poses = np.argsort(final_losses)[:training_hyps_bo['num_best']]
    final_best_states = [final_best_states[pos] for pos in poses]
    final_best_losses = final_losses[poses]
    if problem_type == 'classification':
        final_best_losses_val_accs = [final_best_loss_stats[pos]['best_val_acc'] for pos in poses]
    elif problem_type == 'regression':
        final_best_losses_val_losses = [final_best_loss_stats[pos]['best_val_loss'] for pos in poses]
    final_best_losses_penalizes = [final_best_loss_stats[pos][penalize] for pos in poses]

    total_search_time = time.time() - start_time + prior_time

    printf('\n*---* DRUMROLL... ANNOUNCING BESTS *---*')
    for i, fbs in enumerate(final_best_states):
        if problem_type == 'classification':
            printf('\n#{0}: STATE = {1}, LOSS = {2}, VAL_ACC = {3}, {4} = {5}'.format(i + 1, fbs, final_best_losses[i], final_best_losses_val_accs[i], penalize, final_best_losses_penalizes[i]))
        elif problem_type == 'regression':
            printf('\n#{0}: STATE = {1}, LOSS = {2}, VAL_LOSS = {3}, {4} = {5}'.format(i + 1, fbs, final_best_losses[i], final_best_losses_val_losses[i], penalize, final_best_losses_penalizes[i]))
    printf('\nTOTAL SEARCH TIME = {0} sec = {1} hrs'.format(total_search_time, total_search_time / 3600))

    if problem_type == 'classification':
        final_records = {
            'final_best_states': final_best_states,
            'final_best_losses': final_best_losses,
            'final_best_losses_val_accs': final_best_losses_val_accs,
            'final_best_losses_penalizes': final_best_losses_penalizes,
            'total_search_time': total_search_time / 3600  # in hours
        }
    elif problem_type == 'regression':
        final_records = {
            'final_best_states': final_best_states,
            'final_best_losses': final_best_losses,
            'final_best_losses_val_losses': final_best_losses_val_losses,
            'final_best_losses_penalizes': final_best_losses_penalizes,
            'total_search_time': total_search_time / 3600  # in hours
        }
    with open('./results.pkl', 'wb') as f:
        pickle.dump(final_records, f)
