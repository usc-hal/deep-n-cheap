import time
import numpy as np
import pickle

from .model_search_helper import distancefunc_ramp, kernelfunc_se, dropout, activation, default_weight_decay
from .model_search_helper import bayesopt, downsample, net_kws_defaults, shortcut_conns, get_numparams, run_kws_defaults
from utile import printf


DPCNN_STATE = {
    'out_channels': [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    'embedding_dim': 250,
    'apply_maxpools': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    'shortcuts': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'act': 'relu',
    'strides': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'lr': 0.1, 'weight_decay': 0.0001,
    'batch_size': 64}


def run_model_search_nlp(data, dataset_code,
                         input_size, output_size, problem_type, verbose,
                         wc, tbar_epoch, numepochs, val_patience,
                         bo_prior_states, bo_steps, bo_explore, grid_search_order,
                         num_conv_layers, channels_first, channels_upper, lr, weight_decay, batch_size,
                         bn_fracs, do_fracs, input_drop_probs, drop_probs,
                         num_best, prior_time, embedding_dim, is_dpcnn):

    run_network_kw = {
        'data': data,
        'input_size': input_size,
        'output_size': output_size,
        'problem_type': problem_type,
        'verbose': verbose
    }

    # Covmat params ## # Channels_upper
    distancefunc = distancefunc_ramp
    kernelfunc = kernelfunc_se

    covmat_out_channels_limits = []
    covmat_out_channels_upper_running = channels_first[1]
    for i in range(num_conv_layers[1]):
        covmat_out_channels_limits.append((channels_first[0], covmat_out_channels_upper_running))
        covmat_out_channels_upper_running = np.minimum(covmat_out_channels_upper_running, channels_upper)

    # BO params ##
    out_channels_bo = {'steps': bo_steps, 'initial_states_size': bo_prior_states, 'new_states_size': bo_explore}
    training_hyps_bo = {'steps': bo_steps, 'initial_states_size': bo_prior_states, 'new_states_size': bo_explore, 'num_best': num_best}

    # State params ##
    get_states_limits = {'num_conv_layers': num_conv_layers, 'out_channels_first': channels_first, 'out_channels': (0, channels_upper),
                         'lr': lr, 'weight_decay': weight_decay, 'batch_size': batch_size, 'embedding_dim': embedding_dim}

    start_time = time.time()
    # =============================================================================
    #     Step 1: Bayesian optimization for number of layers and out_channels
    #     Step 2: Fine-tuning architecture using grid search
    #     Step 3: Bayesian optimization for training hyperparameters
    if is_dpcnn:
        the_best_state = DPCNN_STATE
        final_best_states, final_best_loss_stats = step3(get_states_limits, the_best_state, val_patience, numepochs,
                                                         dataset_code, run_network_kw, wc, tbar_epoch, problem_type,
                                                         distancefunc, kernelfunc, training_hyps_bo)
    else:
        the_best_states, the_best_loss_stats = \
            step1(get_states_limits, val_patience, numepochs, dataset_code, run_network_kw,
                  wc, tbar_epoch, problem_type, distancefunc, kernelfunc, covmat_out_channels_limits,
                  embedding_dim, out_channels_bo, start_time, prior_time)
        the_best_loss_val_acc, the_best_loss, the_best_loss_t_epoch, the_best_state = \
            step2(the_best_states, the_best_loss_stats, grid_search_order, val_patience, numepochs,
                  dataset_code, run_network_kw, wc, tbar_epoch, problem_type, do_fracs, input_drop_probs,
                  drop_probs, start_time, prior_time)
        final_best_states, final_best_loss_stats = \
            step3(get_states_limits, the_best_state, val_patience, numepochs,
                  dataset_code, run_network_kw, wc, tbar_epoch, problem_type,
                  distancefunc, kernelfunc, training_hyps_bo)

    # Append architectures ##
    for i in range(len(final_best_states)):
        final_best_states[i] = {**the_best_state, **final_best_states[i]}

    # Calculate number of parameters and default weight decay, then append existing state to final ##
    final_best_states.append({**the_best_state,
                              **{'lr': run_kws_defaults['lr'],
                                 'weight_decay': default_weight_decay(dataset_code=dataset_code, input_size=run_network_kw['input_size'], output_size=run_network_kw['output_size'], net_kw=the_best_state),
                                 'batch_size': run_kws_defaults['batch_size']}
                              })
    final_best_loss_stats.append({'loss': the_best_loss, 'best_val_acc': the_best_loss_val_acc, 't_epoch': the_best_loss_t_epoch})


# =============================================================================
#     Final stats and records
# =============================================================================
    final_losses = np.asarray([fbls['loss'] for fbls in final_best_loss_stats])
    poses = np.argsort(final_losses)[:training_hyps_bo['num_best']]
    final_best_states = [final_best_states[pos] for pos in poses]
    final_best_losses = final_losses[poses]
    final_best_losses_val_accs = [final_best_loss_stats[pos]['best_val_acc'] for pos in poses]
    final_best_losses_t_epochs = [final_best_loss_stats[pos]['t_epoch'] for pos in poses]
    final_numparams = get_numparams(input_size=run_network_kw['input_size'], output_size=run_network_kw['output_size'], net_kw=final_best_states[0])  # all final best states have same architecture, so numparams is just a single value

    total_search_time = time.time() - start_time + prior_time

    printf('\n*---* DRUMROLL... ANNOUNCING BESTS *---*')
    for i, fbs in enumerate(final_best_states):
        printf('\n#{0}: STATE = {1}, LOSS = {2}, VAL_ACC = {3}, T_EPOCH = {4}'.format(i + 1, fbs, final_best_losses[i], final_best_losses_val_accs[i], final_best_losses_t_epochs[i]))
    printf('\nNUM TRAINABLE PARAMETERS = {0}'.format(final_numparams))
    printf('\nTOTAL SEARCH TIME = {0} sec = {1} hrs'.format(total_search_time, total_search_time / 3600))
    final_records = {
        'final_best_states': final_best_states,
        'final_best_losses': final_best_losses,
        'final_best_losses_val_accs': final_best_losses_val_accs,
        'final_best_losses_t_epochs': final_best_losses_t_epochs,
        'final_best_net_numparams': final_numparams,
        'total_search_time': total_search_time / 3600  # in hours
    }

    with open('./results.pkl', 'wb') as f:
        pickle.dump(final_records, f)


def step1(get_states_limits, val_patience, numepochs, dataset_code, run_network_kw,
          wc, tbar_epoch, problem_type, distancefunc, kernelfunc, covmat_out_channels_limits,
          embedding_dim, out_channels_bo, start_time, prior_time):
    printf('STARTING out_channels')
    the_best_states, the_best_loss_stats = bayesopt(
        state_kw={
            'state_keys': ['out_channels', 'embedding_dim'],  # note that only specifying 'out_channels' will also include other channel-dependent keys in each state
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
            'problem_type': problem_type},
        mu_val=None,
        covmat_kw={
            'distancefunc': distancefunc,
            'kernelfunc': kernelfunc,
            'limits': {'out_channels': covmat_out_channels_limits, 'embedding_dim': embedding_dim}},
        cov_keys=['out_channels', 'embedding_dim'],  # specify that only 'out_channels' is being searched over
        **out_channels_bo)
    printf('TOTAL SEARCH TIME = {0}\n\n'.format(time.time() - start_time + prior_time))
    return the_best_states, the_best_loss_stats


def step2(the_best_states, the_best_loss_stats, grid_search_order, val_patience, numepochs,
          dataset_code, run_network_kw, wc, tbar_epoch, problem_type, do_fracs, input_drop_probs,
          drop_probs, start_time, prior_time):
    # Initialization ##
    the_best_state = the_best_states[0]
    the_best_loss = the_best_loss_stats[0]['loss']
    the_best_loss_val_acc = the_best_loss_stats[0]['best_val_acc']
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
                    'problem_type': problem_type})
            if best_loss < the_best_loss:
                the_best_state.update(best_state)
                the_best_loss = best_loss
                the_best_loss_val_acc = loss_stats['best_val_acc']
                the_best_loss_t_epoch = loss_stats['t_epoch']
            else:
                the_best_state.update({'strides': numlayers * net_kws_defaults['strides']})
            printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))

        elif ss == 'do':  # Dropout ##
            printf('STARTING dropout')
            done_state = {'apply_dropouts': (numlayers + 1) * net_kws_defaults['apply_dropouts'],
                          'dropout_probs': [*[net_kws_defaults['dropout_probs'][0]], *(numlayers) * [net_kws_defaults['dropout_probs'][1]]]}
            best_state, best_loss, loss_stats = dropout(
                numlayers=numlayers + 1,
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
                    'problem_type': problem_type},
                done_state=done_state)
            if best_loss < the_best_loss:
                the_best_state.update(best_state)
                the_best_loss = best_loss
                the_best_loss_val_acc = loss_stats['best_val_acc']
            else:
                the_best_state.update(done_state)
            printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))

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
                the_best_loss_t_epoch = loss_stats['t_epoch']
            # no else block since default shortcuts is already in keys
            printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))

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
                the_best_loss_val_acc = loss_stats['best_val_acc']
                the_best_loss_t_epoch = loss_stats['t_epoch']
            else:
                the_best_state.update({'act': 'relu'})
            printf('BEST STATE: {0}, BEST LOSS = {1}, corresponding BEST VAL_ACC = {2} and T_EPOCH = {3}, TOTAL SEARCH TIME = {4}\n\n'.format(the_best_state, the_best_loss, the_best_loss_val_acc, the_best_loss_t_epoch, time.time() - start_time + prior_time))
    return the_best_loss_val_acc, the_best_loss, the_best_loss_t_epoch, the_best_state


def step3(get_states_limits, the_best_state, val_patience, numepochs,
          dataset_code, run_network_kw, wc, tbar_epoch, problem_type,
          distancefunc, kernelfunc, training_hyps_bo):
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
            'problem_type': problem_type
        },
        mu_val=None,
        covmat_kw={
            'distancefunc': distancefunc,
            'kernelfunc': kernelfunc,
            'limits': get_states_limits,  # those are also valid for training hyps
        },
        **training_hyps_bo)
    return final_best_states, final_best_loss_stats
