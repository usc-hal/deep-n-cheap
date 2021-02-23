# =============================================================================
# Model search over low complexity neural networks / cnn-image_classification
# Sourya Dey, USC
# =============================================================================

import time
import numpy as np
import pickle

from model_search.model_search import distancefunc_ramp, kernelfunc_se, dropout, activation, default_weight_decay
from model_search.model_search import bayesopt, downsample, net_kws_defaults, batch_norm, shortcut_conns, get_numparams, run_kws_defaults
from .utile import printf


def run_model_search_cnn(data, dataset_code,
                         input_size, output_size, problem_type, verbose,
                         wc, tbar_epoch, numepochs, val_patience,
                         bo_prior_states, bo_steps, bo_explore, grid_search_order,
                         num_conv_layers, channels_first, channels_upper, lr, weight_decay, batch_size,
                         bn_fracs, do_fracs, input_drop_probs, drop_probs,
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

    covmat_out_channels_limits = []
    covmat_out_channels_upper_running = channels_first[1]   # Limits for # channels in 1st conv layer
    for i in range(num_conv_layers[1]):
        covmat_out_channels_limits.append((channels_first[0], covmat_out_channels_upper_running))
        covmat_out_channels_upper_running = np.minimum(covmat_out_channels_upper_running, channels_upper)

    # BO params ##
    out_channels_bo = {'steps': bo_steps, 'initial_states_size': bo_prior_states, 'new_states_size': bo_explore}
    training_hyps_bo = {'steps': bo_steps, 'initial_states_size': bo_prior_states, 'new_states_size': bo_explore, 'num_best': num_best}

    # State params ##
    get_states_limits = {'num_conv_layers': num_conv_layers, 'out_channels_first': channels_first, 'out_channels': (0, channels_upper),
                         'lr': lr, 'weight_decay': weight_decay, 'batch_size': batch_size}

    start_time = time.time()
# =============================================================================
#     Step 1: Bayesian optimization for number of layers and out_channels
# =============================================================================
    printf('STARTING out_channels')
    the_best_states, the_best_loss_stats = bayesopt(
        state_kw={'state_keys': ['out_channels'],  # note that only specifying 'out_channels' will also include other channel-dependent keys in each state
                  'limits': get_states_limits},
        loss_kw={
            'net_kw_const': {},
            'run_kw_const': {},
            'val_patience': val_patience,
            'numepochs': numepochs,
            'dataset_code': dataset_code,
            'run_network_kw': run_network_kw,
            'wc': wc,
            'tbar_epoch': tbar_epoch,
            'problem_type': problem_type
        },
        mu_val=None,
        covmat_kw={
            'distancefunc': distancefunc,
            'kernelfunc': kernelfunc,
            'limits': {'out_channels': covmat_out_channels_limits},
        },
        cov_keys=['out_channels'],  # specify that only 'out_channels' is being searched over
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
    the_best_loss_t_epoch = the_best_loss_stats[0]['t_epoch']
    numlayers = len(the_best_state['out_channels'])

    # Sub-stages ##
    for ss in grid_search_order:

        if ss == 'ds':  # downsample ##
            printf('STARTING downsampling: strides vs maxpools')
            best_state, best_loss, loss_stats = downsample(
                apply_maxpools=the_best_state['apply_maxpools'],
                loss_kw={
                    'net_kw_const': the_best_state,
                    'run_kw_const': {},
                    'val_patience': val_patience,
                    'numepochs': numepochs,
                    'dataset_code': dataset_code,
                    'run_network_kw': run_network_kw,
                    'wc': wc,
                    'tbar_epoch': tbar_epoch,
                    'problem_type': problem_type
                }
            )
            if best_loss < the_best_loss:
                the_best_state.update(best_state)
                the_best_loss = best_loss
                if problem_type == 'classification':
                    the_best_loss_val_acc = loss_stats['best_val_acc']
                elif problem_type == 'regression':
                    the_best_loss_val_loss = loss_stats['best_val_loss']
                the_best_loss_t_epoch = loss_stats['t_epoch']
            else:
                the_best_state.update({'strides': numlayers * net_kws_defaults['strides']})
            if problem_type == 'classification':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))
            elif problem_type == 'regression':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_LOSS = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_loss, the_best_loss_t_epoch, time.time() - start_time + prior_time))

        elif ss == 'bn':  # BN ##
            printf('STARTING batch norm')
            best_state, best_loss, loss_stats = batch_norm(
                numlayers=numlayers,
                fracs=bn_fracs,
                loss_kw={
                    'net_kw_const': the_best_state,
                    'run_kw_const': {},
                    'val_patience': val_patience,
                    'numepochs': numepochs,
                    'dataset_code': dataset_code,
                    'run_network_kw': run_network_kw,
                    'wc': wc,
                    'tbar_epoch': tbar_epoch,
                    'problem_type': problem_type
                }
            )
            if best_loss < the_best_loss:
                the_best_state.update(best_state)
                the_best_loss = best_loss
                if problem_type == 'classification':
                    the_best_loss_val_acc = loss_stats['best_val_acc']
                elif problem_type == 'regression':
                    the_best_loss_val_loss = loss_stats['best_val_loss']
                the_best_loss_t_epoch = loss_stats['t_epoch']
            else:
                the_best_state.update({'apply_bns': numlayers * net_kws_defaults['apply_bns']})
            if problem_type == 'classification':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))
            elif problem_type == 'regression':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_LOSS = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_loss, the_best_loss_t_epoch, time.time() - start_time + prior_time))

        elif ss == 'do':  # Dropout ##
            printf('STARTING dropout')
            done_state = {'apply_dropouts': numlayers * net_kws_defaults['apply_dropouts'],
                          'dropout_probs': [*[net_kws_defaults['dropout_probs'][0]],
                                            *(numlayers - 1) * [net_kws_defaults['dropout_probs'][1]]]}
            best_state, best_loss, loss_stats = dropout(
                numlayers=numlayers,
                fracs=do_fracs,
                input_drop_probs=input_drop_probs,
                drop_probs=drop_probs,
                loss_kw={
                    'net_kw_const': the_best_state,
                    'run_kw_const': {},
                    'val_patience': val_patience,
                    'numepochs': numepochs,
                    'dataset_code': dataset_code,
                    'run_network_kw': run_network_kw,
                    'wc': wc,
                    'tbar_epoch': tbar_epoch,
                    'problem_type': problem_type
                },
                done_state=done_state)
            if best_loss < the_best_loss:
                the_best_state.update(best_state)
                the_best_loss = best_loss
                if problem_type == 'classification':
                    the_best_loss_val_acc = loss_stats['best_val_acc']
                elif problem_type == 'regression':
                    the_best_loss_val_loss = loss_stats['best_val_loss']
                the_best_loss_t_epoch = loss_stats['t_epoch']
            else:
                the_best_state.update(done_state)
            if problem_type == 'classification':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))
            elif problem_type == 'regression':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_LOSS = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_loss, the_best_loss_t_epoch, time.time() - start_time + prior_time))

        elif ss == 'sc':  # Shortcuts ##
            printf('STARTING shortcuts')
            best_state, best_loss, loss_stats = shortcut_conns(
                numlayers=numlayers,
                loss_kw={
                    'net_kw_const': the_best_state,
                    'run_kw_const': {},
                    'val_patience': val_patience,
                    'numepochs': numepochs,
                    'dataset_code': dataset_code,
                    'run_network_kw': run_network_kw,
                    'wc': wc,
                    'tbar_epoch': tbar_epoch,
                    'problem_type': problem_type})

            if best_loss < the_best_loss:
                the_best_state.update(best_state)
                the_best_loss = best_loss
                if problem_type == 'classification':
                    the_best_loss_val_acc = loss_stats['best_val_acc']
                elif problem_type == 'regression':
                    the_best_loss_val_loss = loss_stats['best_val_loss']
                the_best_loss_t_epoch = loss_stats['t_epoch']
            # no else block since default shortcuts is already in keys
            if problem_type == 'classification':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))
            elif problem_type == 'regression':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_LOSS = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_loss, the_best_loss_t_epoch, time.time() - start_time + prior_time))

        elif ss == 'ac':  # activation ##
            printf('STARTING activation')
            best_state, best_loss, loss_stats = activation(
                numlayers=numlayers,
                loss_kw={
                    'net_kw_const': the_best_state,
                    'run_kw_const': {},
                    'val_patience': val_patience,
                    'numepochs': numepochs,
                    'dataset_code': dataset_code,
                    'run_network_kw': run_network_kw,
                    'wc': wc,
                    'tbar_epoch': tbar_epoch,
                    'problem_type': problem_type})

            if best_loss < the_best_loss:
                the_best_state.update(best_state)
                the_best_loss = best_loss
                if problem_type == 'classification':
                    the_best_loss_val_acc = loss_stats['best_val_acc']
                elif problem_type == 'regression':
                    the_best_loss_val_loss = loss_stats['best_val_loss']
                the_best_loss_t_epoch = loss_stats['t_epoch']
            else:
                the_best_state.update({'act': 'relu'})
            if problem_type == 'classification':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))
            elif problem_type == 'regression':
                printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))


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
            'wc': wc,
            'tbar_epoch': tbar_epoch,
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
                                 'weight_decay': default_weight_decay(dataset_code=dataset_code, input_size=run_network_kw['input_size'],
                                                                      output_size=run_network_kw['output_size'], net_kw=the_best_state),
                                 'batch_size': run_kws_defaults['batch_size']}})
    if problem_type == 'classification':
        final_best_loss_stats.append({'loss': the_best_loss, 'best_val_acc': the_best_loss_val_acc, 't_epoch': the_best_loss_t_epoch})
    elif problem_type == 'regression':
        final_best_loss_stats.append({'loss': the_best_loss, 'best_val_loss': the_best_loss_val_loss, 't_epoch': the_best_loss_t_epoch})

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
    final_best_losses_t_epochs = [final_best_loss_stats[pos]['t_epoch'] for pos in poses]
    final_numparams = get_numparams(input_size=run_network_kw['input_size'], output_size=run_network_kw['output_size'], net_kw=final_best_states[0])  # all final best states have same architecture, so numparams is just a single value

    total_search_time = time.time() - start_time + prior_time

    printf('\n*---* DRUMROLL... ANNOUNCING BESTS *---*')
    for i, fbs in enumerate(final_best_states):
        if problem_type == 'classification':
            printf('\n#{0}: STATE = {1}, LOSS = {2}, VAL_ACC = {3}, T_EPOCH = {4}'.format(i + 1, fbs, final_best_losses[i], final_best_losses_val_accs[i], final_best_losses_t_epochs[i]))
        elif problem_type == 'regression':
            printf('\n#{0}: STATE = {1}, LOSS = {2}, VAL_LOSS = {3}, T_EPOCH = {4}'.format(i + 1, fbs, final_best_losses[i], final_best_losses_val_losses[i], final_best_losses_t_epochs[i]))
    printf('\nNUM TRAINABLE PARAMETERS = {0}'.format(final_numparams))
    printf('\nTOTAL SEARCH TIME = {0} sec = {1} hrs'.format(total_search_time, total_search_time / 3600))

    if problem_type == 'classification':
        final_records = {
            'final_best_states': final_best_states,
            'final_best_losses': final_best_losses,
            'final_best_losses_val_accs': final_best_losses_val_accs,
            'final_best_losses_t_epochs': final_best_losses_t_epochs,
            'final_best_net_numparams': final_numparams,
            'total_search_time': total_search_time / 3600  # in hours
        }
    elif problem_type == 'regression':
        final_records = {
            'final_best_states': final_best_states,
            'final_best_losses': final_best_losses,
            'final_best_losses_val_losses': final_best_losses_val_losses,
            'final_best_losses_t_epochs': final_best_losses_t_epochs,
            'final_best_net_numparams': final_numparams,
            'total_search_time': total_search_time / 3600  # in hours
        }
    with open('./results.pkl', 'wb') as f:
        pickle.dump(final_records, f)
