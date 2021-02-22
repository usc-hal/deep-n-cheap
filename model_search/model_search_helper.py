# =============================================================================
# Model search over low complexity neural networks
# Sourya Dey, USC
# =============================================================================

import numpy as np
from scipy import linalg
from scipy.stats import norm as gaussian
import sobol_seq
import itertools
from model.model import run_network, get_numparams, net_kws_defaults, run_kws_defaults, nn_activations
from utile import printf


# =============================================================================
# Helper functions
# =============================================================================
def convert_keys(state_keys):
    ''' Convert state keys to covmat keys '''
    cov_keys = []
    for key in state_keys:
        if key in ['out_channels', 'hidden_mlp', 'lr', 'weight_decay', 'batch_size']:
            cov_keys.append(key)
    return cov_keys


def default_weight_decay(dataset_code, input_size, output_size, net_kw):
    ''' Get default weight decay for any net '''
    numparams = get_numparams(input_size, output_size, net_kw)
    if len(input_size) > 1:  # any CNN dataset
        dwd = numparams * 1e-11 if numparams >= 1e6 else 0.  # number of parameters in M * 1e-5. Eg: 0 for numparams < 1M, 1e-4 for numparams = 10M, etc
    elif 'F' in dataset_code or 'M' in dataset_code:
        dwd = numparams * 1e-9 if numparams >= 1e4 else 0.
    elif 'R' in dataset_code:
        dwd = numparams * 1e-10 if numparams >= 1e5 else 0.
    elif dataset_code == "Y" or dataset_code == "S":
        dwd = 1e-4
    else:  # custom MLP dataset
        dwd = 0.
    return dwd


def form_shortcuts_start_every(numlayers):
    return np.inf if numlayers <= 2 else 2  # Sara


def form_shortcuts(num_conv_layers, start_from=0, start_every=2):
    '''
    start_every: Start the next shortcut this many layers after the previous shortcut started
    Example: num_conv_layers = 16
        start_every = 2 --> [1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 0,0]
        start_every = 4 --> [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]
    start_every = np.inf means no shortcuts, i.e. shortcuts = all 0s
    '''
    shortcuts = num_conv_layers * [0]
    if not start_every == np.inf:
        for i in range(start_from, num_conv_layers - 2, start_every):
            shortcuts[i] = 1
    return shortcuts
# =============================================================================


# =============================================================================
# Get state
# =============================================================================
def get_states(numstates=15,
               state_keys=['out_channels'],
               samp='sobol', nsv=20,
               limits={'num_conv_layers': (4, 8), 'out_channels_first': (16, 64),
                       'out_channels': (0, 512), 'num_hidden_layers_mlp': (0, 3),
                       'hidden_nodes_mlp': (50, 1000), 'lr': (-5, -1),
                       'weight_decay': (-6, -3), 'batch_size': (32, 512)}):
    '''
    *** Generate a number of states ***
    numstates: How many states to generate
    state_keys: Each state is a dict with these keys, i.e. these form the current search space
    samp: 'sobol' to use Sobol sampling, 'random' to use random sampling
    nsv: Number of sampling vectors. Each vector is unique. Ideally this should be greater than number of keys, but even if not, vectors are repeated cyclically
        Eg: Say search space has out_channels and max num_conv_layers is 10. Then nsv should ideally be >=10
    limits: Limits for different state_keys. Keys may not match those in state_keys
        out_channels lower limit is a dummy. Upper limit is important
        lr limits are log10(lr)
        weight_decay limits are log10(wd), and the last power of 10 is converted to actual weight_decay = 0
    '''
    states = [{key: [] for key in state_keys} for _ in range(numstates)]
    if samp == 'sobol':
        samp = sobol_seq.i4_sobol_generate(nsv, numstates)  # output array will be of shape (numstates,nsv)
    elif samp == 'random':
        samp = np.random.rand(numstates, nsv)
    si = np.random.randint(nsv)  # samp column index. pick a random place to start from

    # out_channels ##
    if 'out_channels' in state_keys:
        # num_conv_layers ##
        lower, upper = limits['num_conv_layers']
        num_conv_layers = (lower + samp[:, si % nsv] * (upper + 1 - lower)).astype('int')  # casting to int is flooring, hence doing upper+1
        si += 1

        # out_channels_first ##
        lower, upper = limits['out_channels_first']
        out_channels_first = (lower + samp[:, si % nsv] * (upper + 1 - lower)).astype('int')
        lower, upper = limits['embedding_dim']
        embedding_dim = (lower + samp[:, si % nsv] * (upper + 1 - lower)).astype('int')
        # don't increment si right now, it will be done after all out_channels are done

        # remaining out_channels, and other channel-dependent keys like downsampling and shortcuts ##
        for n in range(numstates):
            states[n]['out_channels'] = num_conv_layers[n] * [0]
            states[n]['out_channels'][0] = out_channels_first[n]
            states[n]['apply_maxpools'] = num_conv_layers[n] * [0]
            states[n]['embedding_dim'] = embedding_dim[n]
            # count_maxpools = 0

            for i in range(1, num_conv_layers[n]):
                lower = states[n]['out_channels'][i - 1]
                upper = np.minimum(2 * states[n]['out_channels'][i - 1], limits['out_channels'][1])
                states[n]['out_channels'][i] = (lower + samp[n, (si + i) % nsv] * (upper + 1 - lower)).astype('int')
                # if states[n]['out_channels'][i] > 64 and count_maxpools == 0:
                #     states[n]['apply_maxpools'][i - 1] = 1
                #     count_maxpools += 1
                # elif states[n]['out_channels'][i] > 128 and count_maxpools == 1:
                #     states[n]['apply_maxpools'][i - 1] = 1
                #     count_maxpools += 1
                # elif states[n]['out_channels'][i] > 256 and count_maxpools == 2:
                #     states[n]['apply_maxpools'][i - 1] = 1
                #     count_maxpools += 1
                if states[n]['apply_maxpools'][i - 1] == 0:
                    states[n]['apply_maxpools'][i] = np.random.randint(2)

            # states[n]['shortcuts'] = form_shortcuts(num_conv_layers[n], start_from=1, start_every=form_shortcuts_start_every(num_conv_layers[n]))
            # change :
            states[n]['shortcuts'] = form_shortcuts(num_conv_layers[n], start_every=form_shortcuts_start_every(num_conv_layers[n]))

        si += limits['num_conv_layers'][1]

    # hidden_mlp ##
    if 'hidden_mlp' in state_keys:
        # num_hidden_layers_mlp ##
        lower, upper = limits['num_hidden_layers_mlp']
        num_hidden_layers_mlp = (lower + samp[:, si % nsv] * (upper + 1 - lower)).astype('int')  # casting to int is flooring, hence doing upper+1
        si += 1

        # hidden_nodes ##
        lower, upper = limits['hidden_nodes_mlp']
        hidden_mlps = []
        for _ in range(limits['num_hidden_layers_mlp'][1]):
            hidden_mlps.append((lower + samp[:, si % nsv] * (upper + 1 - lower)).astype('int'))  # single layer for all states
            si += 1
        hidden_mlps = np.asarray(hidden_mlps)  # shape = (upper_limit_num_hidden_layers_mlp, numstates)
        for n in range(numstates):
            states[n]['hidden_mlp'] = list(hidden_mlps[:num_hidden_layers_mlp[n], n])

    # lr ##
    if 'lr' in state_keys:
        lower, upper = limits['lr']
        lrs = 10 ** (lower + samp[:, si % nsv] * (upper - lower))
        si += 1
        for n in range(numstates):
            states[n]['lr'] = lrs[n]
        # states[0]['lr'] = 0.001
        # for n in range(1, numstates):
        #     states[n]['lr'] = lrs[n]

    # weight_decay ##
    if 'weight_decay' in state_keys:
        lower, upper = limits['weight_decay']
        weight_decays = 10 ** (lower + samp[:, si % nsv] * (upper - lower))
        si += 1
        weight_decays[weight_decays < 10**(lower + 1)] = 0.  # make lowest order of magnitude = 0
        for n in range(numstates):
            states[n]['weight_decay'] = weight_decays[n]
        # states[0]['weight_decay'] = 0
        # for n in range(1, numstates):
        #     states[n]['weight_decay'] = weight_decays[n]

    # batch size ##
    if 'batch_size' in state_keys:
        lower, upper = limits['batch_size']
        batch_sizes = (lower + samp[:, si % nsv] * (upper + 1 - lower)).astype(int)
        si += 1
        for n in range(numstates):
            states[n]['batch_size'] = batch_sizes[n].item()  # .item() is required to convert to native Python int, which is required by Pytorch

    return states


# =============================================================================
# Loss function
# =============================================================================
def lossfunc(state, net_kw_const={}, run_kw_const={},
             validate=True, val_patience=np.inf, test=True, numepochs=100,
             dataset_code='T', run_network_kw={}, penalize='t_epoch',
             wc=0.1, tbar_epoch=1, numparams_bar=15e6, problem_type='classification'):
    '''
    *** Wrapper function for run_network. Given a net, find its model search loss and other statistics ***

    Net is described using:
        state : Params being optimized over
        net_kw_const, run_kw_const : Params not being optimized over
    These combine to form net_kw and run_kw for run_network

    Parameters of run_network which might change according to lossfunc are described using:
        validate, val_patience, test, numepochs

    Other parameters of run_network which are not expected to change are given in run_network_kw
        Example: data, input_size, output_size, num_workers, pin_memory, wt_init, bias_init, verbose

    penalize: Either 't_epoch' or 'numparams' (CNNs only support t_epoch)
    wc : weightage given to complexity term
    tbar_epoch : used to normalize training time
    numparams_bar: Used to normalize number of parameters

    Returns: loss_stats dictionary. Most important key is 'loss', which gives model search loss
    '''
    net_kw = {**net_kw_const, **{key: state[key] for key in state.keys() if key in net_kws_defaults}}
    run_kw = {**run_kw_const, **{key: state[key] for key in state.keys() if key in run_kws_defaults}}
    if 'weight_decay' not in state.keys():
        run_kw['weight_decay'] = default_weight_decay(dataset_code=dataset_code, input_size=run_network_kw['input_size'], output_size=run_network_kw['output_size'], net_kw=net_kw)

    net, recs = run_network(
        net_kw=net_kw, run_kw=run_kw, validate=validate,
        val_patience=val_patience, test=test, numepochs=numepochs,
        **run_network_kw)

    # Find model search stats ##
    numparams = get_numparams(input_size=run_network_kw['input_size'], output_size=run_network_kw['output_size'], net_kw=net_kw)
    loss_stats = {}

    if problem_type == 'classification':
        acc, _ = np.max(recs['val_accs']), np.argmax(recs['val_accs']) + 1  # recs['val_accs'].max(0)
        fp = (100 - acc) / 100.0
        loss_stats['best_val_acc'] = np.max(recs['val_accs'])  # torch.max(recs['val_accs']).item()
    elif problem_type == 'regression':
        loss, _ = np.min(recs['val_losses']), np.argmin(recs['val_losses']) + 1
        scale = 10  # TODO: tune this scale factor
        fp = loss * scale
        loss_stats['best_val_loss'] = np.min(recs['val_losses'])

    fc = recs['t_epoch'] / tbar_epoch if penalize == 't_epoch' else numparams / numparams_bar
    loss_stats['loss'] = np.log10(fp + wc * fc)

    loss_stats['t_epoch'] = recs['t_epoch']
    loss_stats['numparams'] = numparams

    return loss_stats


def distancefunc_ramp(x1, x2, omega, upper, lower, root=1):
    '''
    Our own distance function
    Ranges:
        omega >= 0
        root = 1,2,3,4
    '''
    if upper == lower:
        return 0.
    else:
        return omega * (np.abs(x1 - x2) / (upper - lower))**(1 / root)


def kernelfunc_se(d):
    ''' Convert a single scalar distance value d to a kernel value using squared exponential '''
    return np.exp(-0.5 * d**2)


# =============================================================================
# Covariance matrix
# =============================================================================
def covmat(S1, S2,
           distancefunc, kernelfunc, limits,
           cov_keys, omega, kernel_comb_weights):
    '''
    *** Find and return covariance matrix of shape (len(S1),len(S2)) ***

    S1, S2 : Lists of states, each state is a dictionary, must have same keys

    distancefunc, kernelfunc : Choose from previously defined functions
    limits : Upper and lower limits for all covmat keys.
        THIS IS DIFFERENT from limits in get_states. Those were for out_channels_first, etc

    omega, kernel_comb_weights : Kernel hyperparameters to be optimized for all covmat keys
        Max distance is omega for ramp and 2*omega for hutter, while min distance is 0. So min kernel value is exp(-0.5 * (2*omega)**2) for hutter, while max kernel value is exp(0) = 1
        kernel_comb_weights are coefficients to combine individual kernel values for different parameters. Will be normalized inside
    WHY USE THESE HYPS?
        Having these is equivalent to using full version of SE kernel, which is sigma1**2 * exp(-0.5 * (d1/l1)**2) + sigma2**2 * exp(-0.5 * (d2/l2)**2) + ...
        Here, omega is proportional to 1/l and kernel_comb_weight to sigma**2
    '''
    # Find sum of all kernel_comb_weights across all keys ##
    kernel_comb_norm = sum([v for v in kernel_comb_weights.values()])

    K = np.zeros((len(S1), len(S2)), dtype='float')
    for ind1, s1 in enumerate(S1):
        for ind2, s2 in enumerate(S2):

            # Trivial case ##
            if s1 == s2:
                K[ind1, ind2] = 1.

            # Regular case ##
            else:
                for key in cov_keys:

                    # Convert state values to covmat values ##
                    if key == 'out_channels':
                        if len(s1[key]) >= len(s2[key]):
                            sbig, ssmall = s1, s2
                        else:
                            sbig, ssmall = s2, s1
                        kern = 0.
                        for i in range(len(sbig[key])):
                            root = np.minimum(i + 1, 4)  # 1 for 1st layer, square root for 2nd, cube root for 3rd, 4th root for 4th and beyond layers. Do this because different layers have different ranges, and we want to treat them the same way
                            if i < len(ssmall[key]):
                                d = distancefunc_ramp(sbig[key][i], ssmall[key][i], omega=omega[key], upper=limits[key][i][1], lower=limits[key][i][0], root=root)
                            else:
                                d = omega[key]
                            kern += kernelfunc(d) / len(sbig[key])

                    elif key == 'hidden_mlp' or key == 'embedding_dim':
                        kern = kernelfunc(distancefunc(np.sum(s1[key]), np.sum(s2[key]), omega=omega[key], upper=limits[key][1], lower=limits[key][0], root=2))
                        kern += kernelfunc(distancefunc(1, 1, omega=omega[key], upper=1, lower=1))
                        kern /= 2

                    else:
                        # Single value
                        if key == 'batch_size':
                            val1, val2 = s1[key], s2[key]

                        # Single value in log space
                        elif key == 'lr':
                            val1, val2 = np.log10(s1[key]), np.log10(s2[key])

                        # Single value in log space, can be 0. Treat 0 as the lowest value so that we can take log
                        elif key == 'weight_decay':
                            val1 = limits[key][0] if s1[key] == 0 else np.log10(s1[key])
                            val2 = limits[key][0] if s2[key] == 0 else np.log10(s2[key])

                        # Find distance and kernel ##
                        d = distancefunc(val1, val2, omega=omega[key], upper=limits[key][1], lower=limits[key][0])
                        kern = kernelfunc(d)

                    # Construct overall kernel ##
                    K[ind1, ind2] += (kernel_comb_weights[key] / kernel_comb_norm * kern)
    return K

# =============================================================================
# Bayesian optimization
# =============================================================================


def gp_predict(new_S, S, mu_val, norm_Y, K_inv, **covmat_kw):
    '''
    *** Given prior GP and new points, predict posterior mean and covariance matrix ***
    new_S : List of states of length n
    S : Prior states of length m
    mu_val : Scalar value to fill mean vector
    norm_Y : Normalized (mean subtracted) prior outputs of length m
    K_inv : (Noise added) inverse of prior cov matrix, size m,m
    **covmat_kw : All else needed for cov matrix, like distancefunc, kernelfunc, limits, cov_keys, omega, rho, kernel_comb_weights
    '''
    new_mu = mu_val * np.ones(len(new_S))  # (n,)
    new_K = covmat(S1=new_S, S2=new_S, **covmat_kw)  # Brochu tutorial recommends not adding noise_var here, otherwise do +noise_var*np.eye(len(new_S))
    new_old_K = covmat(S1=new_S, S2=S, **covmat_kw)  # Do not add noise_var here since new_old_K is not diagonal (since in general n != m)
    post_mu = new_mu + new_old_K @ K_inv @ norm_Y  # (n,)
    post_cov = new_K - new_old_K @ K_inv @ new_old_K.T  # (n,n)
    return post_mu, post_cov


def ei(post_mu, post_std, best_Y, ksi=0.01):
    '''
    *** Given SCALAR posterior mean and standard deviation of a SINGLE new point, find its expected improvement ***
    best_Y : Current best value based on previous points
    ksi : Exploration-exploitation tradeoff, see Sec 2.4 of https://arxiv.org/pdf/1012.2599.pdf
    '''
    imp = best_Y - post_mu - ksi  # improvement
    Z = imp / post_std if post_std > 0 else 0
    ei = imp * gaussian.cdf(Z) + post_std * gaussian.pdf(Z)  # single number
    return ei


def bayesopt(state_kw={}, loss_kw={},
             mu_val=None, covmat_kw={}, cov_keys=[],
             omega={}, kernel_comb_weights={}, noise_var=1e-4,
             steps=20, initial_states_size=10, new_states_size=100, ksi=1e-4,
             num_best=1):
    '''
    state_kw : kwargs for get_state()
    loss_kw : kwargs for lossfunc()
    mu_val : If None, always normalize values to get mu for all. Otherwise make mu for all equal to this
    covmat_kw : kwargs for covmat. Do NOT omit any kwargs here since these are pased as args when calling minimize()

    steps : #steps in BO
    initial_states_size : #prior points to form initial approximation
    new_states_size : #points for which to calculate acquisition function in each step
    ksi : As in ei()

    num_best : Return this many best states, e.g. top 3
    '''
# =============================================================================
#     Get initial states and losses
# =============================================================================
    printf('{0} initial states:'.format(initial_states_size))
    states = get_states(numstates=initial_states_size, samp='sobol', **state_kw)
    loss_stats = []
    losses = np.asarray([])
    for i, state in enumerate(states):
        loss_stats.append(lossfunc(state=state, **loss_kw))
        losses = np.append(losses, loss_stats[-1]['loss'])
        printf('State {0} = {1}, Loss = {2}\n'.format(i + 1, state, losses[-1]))

# =============================================================================
#     Optimize
# =============================================================================
    printf('Optimization starts:')
    cov_keys = convert_keys(states[0].keys()) if cov_keys == [] else cov_keys

    # Set kernel hyperparameters (if they are being optimized using ML, they will be overwritten later) ##
    omega = {key: 3 for key in cov_keys} if omega == {} else omega
    kernel_comb_weights = {key: 1 for key in cov_keys} if kernel_comb_weights == {} else kernel_comb_weights   # Sara
    # kernel_comb_weights = {key: 1 for key in cov_keys} if kernel_comb_weights == {} else kernel_comb_weights   # Sara

    covmat_kw.update({'cov_keys': cov_keys, 'omega': omega, 'kernel_comb_weights': kernel_comb_weights})
    for step in range(steps):

        # Form optimal GP from existing states ##
        mu_val = np.mean(losses) if mu_val is None else mu_val
        mu = mu_val * np.ones(len(losses))  # vectorize
        K = covmat(S1=states, S2=states, **covmat_kw)

        # Get stuff needed to calculate acquisition function ##
        K_inv = linalg.inv(K + noise_var * np.eye(K.shape[0]))
        norm_losses = losses - mu

        # ### Since samples are expected to be noisy, use the best expected loss instead of the actual loss (see Sec 2.4 of https://arxiv.org/pdf/1012.2599.pdf). But I tried this and it gives worse results, so not doing ####
        # best_losses_expec = np.min(gp_predict(new_S=states, S=states, mu_val=mu_val, norm_Y=norm_losses, K_inv=K_inv, **covmat_kw)[0])

        # Find new best state via acquisition function ##
        new_states = get_states(numstates=new_states_size, samp='random', **state_kw)  # don't use sobol sampling here since that will revisit states
        eis = np.zeros(len(new_states))
        for i in range(len(new_states)):
            post_mu, post_var = gp_predict(new_S=new_states[i: i + 1], S=states, mu_val=mu_val, norm_Y=norm_losses, K_inv=K_inv, **covmat_kw)
            eis[i] = ei(post_mu=post_mu.flatten()[0], post_std=np.sqrt(post_var.flatten()[0]), best_Y=np.min(losses), ksi=ksi)
            # eis[i] = ei(post_mu = post_mu.flatten()[0], post_std = np.sqrt(post_var.flatten()[0]), best_Y = best_losses_expec, ksi = ksi) #If best_losses_expec is used
        best_state = new_states[np.argmax(eis)]  # find state with highest EI

        # Add attributes of best state to existing attributes ##
        states.append(best_state)
        loss_stats.append(lossfunc(state=best_state, **loss_kw))
        losses = np.append(losses, loss_stats[-1]['loss'])
        printf('Step {0}, Best State = {1}, Loss = {2}\n'.format(step + 1, best_state, losses[-1]))

    # Final optimization results ##
    poses = np.argsort(losses)
    best_states = [states[pos] for pos in poses[:num_best]]
    best_loss_stats = [loss_stats[pos] for pos in poses[:num_best]]
    for i, bs in enumerate(best_states):
        printf('#{0}: Best state = {1}, Corresponding stats = {2}'.format(i + 1, bs, best_loss_stats[i]))
    return best_states, best_loss_stats


# =============================================================================
# Other search methods
# =============================================================================
def downsample(apply_maxpools, loss_kw={}):
    '''
    Compare strides vs max-pooling in the locations specified by 1s in apply_maxpools
    Eg: If apply_maxpools = [0,1,1,0], then compare:
        strides = [1,2,2,1], apply_maxpools = [0,0,0,0]
        strides = [1,2,1,1], apply_maxpools = [0,0,1,0]
        strides = [1,1,2,1], apply_maxpools = [0,1,0,0]
        ALREADY DONE: strides = [1,1,1,1], apply_maxpools = [0,1,1,0]
    '''
    states = []
    loss_stats = []
    losses = np.asarray([])
    numlayers = len(apply_maxpools)
    locs = np.where(np.asarray(apply_maxpools) == 1)[0]  # where to insert downsampling

    if len(locs) > 0:
        for bdigits in itertools.product([0, 1], repeat=len(locs)):  # 0s are strides, 1s are maxpools

            # Last case is all 1s, i.e. only maxpools, no strides. This case is already done during out_channels search hence skip it
            if sum(bdigits) == len(locs):
                continue

            # Regular case with strides
            strides = numlayers * [1]
            apply_maxpools = numlayers * [0]
            for i in range(len(bdigits)):
                if bdigits[i] == 0:
                    strides[locs[i]] = 2
                    # pass
                else:
                    apply_maxpools[locs[i]] = 1

            # Now run network
            states.append({'strides': strides, 'apply_maxpools': apply_maxpools})
            loss_stats.append(lossfunc(state=states[-1], **loss_kw))
            losses = np.append(losses, loss_stats[-1]['loss'])
            printf('State = {0}, Loss = {1}\n'.format(states[-1], losses[-1]))

        best_pos, best_loss = np.argmin(losses), np.min(losses)
        best_state, best_loss_stats = states[best_pos], loss_stats[best_pos]
        printf('\nBest state = {0}, Best loss = {1}'.format(best_state, best_loss))
        return best_state, best_loss, best_loss_stats

    else:
        printf('No downsampling cases to try')
        return None, np.inf, None


def batch_norm(numlayers, fracs=[0, 0.25, 0.5, 0.75], loss_kw={}):
    '''
    Batch norm all 1 is already done
    Here run for number of batch norm layers = fracs * number of total layers
    BN layers are kept as late as possible
    Example: numlayers = 7, frac = 0.5
        #BN layers = 7*0.5 rounded up = 4
        7/4 = 1.75, so BN layers should come after layer 1.75, 3.5, 5.25, 7
        Rounding up gives 2, 4, 6, 7, i.e. apply_bns = [0,1,0,1,0,1,1]
    '''
    states = []
    loss_stats = []
    losses = np.asarray([])

    for frac in fracs:
        if frac == 1:
            continue  # already done

        apply_bns = np.zeros(numlayers)

        if frac != 0:
            num_bns = int(np.ceil(frac * numlayers))
            intervals = np.arange(numlayers / num_bns, numlayers + 0.001, numlayers / num_bns, dtype='half')  # use dtype=half to avoid numerical issues
            intervals = np.ceil(intervals).astype('int')
            apply_bns[intervals - 1] = 1

        states.append({'apply_bns': [int(x) for x in apply_bns]})
        loss_stats.append(lossfunc(state=states[-1], **loss_kw))
        losses = np.append(losses, loss_stats[-1]['loss'])
        printf('State = {0}, Loss = {1}\n'.format(states[-1], losses[-1]))

    best_pos, best_loss = np.argmin(losses), np.min(losses)
    best_state, best_loss_stats = states[best_pos], loss_stats[best_pos]
    printf('\nBest state = {0}, Best loss = {1}'.format(best_state, best_loss))
    return best_state, best_loss, best_loss_stats


def activation(numlayers, loss_kw={}):
    '''
    Apply chosen activation function in every layer
    Example: numlayers = 7, nn_activations.keys() = ['relu', 'tanh', 'sigmoid']
        1st search: use relu in every layer
        2nd search: use tanh in every layer
        3rd search: use sigmoid in every layer
    '''
    states = []
    loss_stats = []
    losses = np.asarray([])

    for act in nn_activations.keys():
        states.append({'act': act})
        loss_stats.append(lossfunc(state=states[-1], **loss_kw))
        losses = np.append(losses, loss_stats[-1]['loss'])
        printf('State = {0}, Loss = {1}\n'.format(states[-1], losses[-1]))

    best_pos, best_loss = np.argmin(losses), np.min(losses)
    best_state, best_loss_stats = states[best_pos], loss_stats[best_pos]
    printf('\nBest state = {0}, Best loss = {1}'.format(best_state, best_loss))
    return best_state, best_loss, best_loss_stats


def dropout(numlayers, fracs=[0, 0.25, 0.5, 0.75, 1], input_drop_probs=[0.1, 0.2], drop_probs=[0.15, 0.3, 0.45], loss_kw={}, done_state={}):
    '''
    Same logic as batch_norm, i.e. distribute frac number of dropout layers evenly (as late as possible for fractions)
    For each dropout config, all intermediate layer drop probs are kept to p which varies in drop_probs
    If input layer dropout is present, its keep prob p varies in input_drop_probs
    done_state is any state which has already been tested, example for numlayers = 4, done_state could be {'apply_dropouts':[1,1,1,1], 'dropout_probs':[0.1,0.3,0.3,0.3]}
    '''
    states = []
    loss_stats = []
    losses = np.asarray([])

    for frac in fracs:
        apply_dropouts = np.zeros(numlayers)
        dropout_probs = []

        if frac == 0:
            states.append({'apply_dropouts': [int(x) for x in apply_dropouts], 'dropout_probs': dropout_probs})

        else:
            num_dropouts = int(np.ceil(frac * numlayers))
            intervals = np.arange(numlayers / num_dropouts, numlayers + 0.001, numlayers / num_dropouts, dtype='half')  # use dtype=half to avoid numerical issues
            intervals = np.ceil(intervals).astype('int')
            apply_dropouts[intervals - 1] = 1

            for drop_prob in drop_probs:
                if apply_dropouts[0] != 1:
                    dropout_probs = num_dropouts * [drop_prob]
                    states.append({'apply_dropouts': [int(x) for x in apply_dropouts], 'dropout_probs': dropout_probs})
                else:
                    for input_drop_prob in input_drop_probs:
                        dropout_probs = [* [input_drop_prob], * (num_dropouts - 1) * [drop_prob]]
                        states.append({'apply_dropouts': [int(x) for x in apply_dropouts], 'dropout_probs': dropout_probs})

    for i, state in enumerate(states):
        if list(state['apply_dropouts']) == list(done_state['apply_dropouts']) and list(state['dropout_probs']) == list(done_state['dropout_probs']):
            del states[i]  # if current state is the same as done state, delete it
    for state in states:
        loss_stats.append(lossfunc(state=state, **loss_kw))
        losses = np.append(losses, loss_stats[-1]['loss'])
        printf('State = {0}, Loss = {1}\n'.format(state, losses[-1]))

    best_pos, best_loss = np.argmin(losses), np.min(losses)
    best_state, best_loss_stats = states[best_pos], loss_stats[best_pos]
    printf('\nBest state = {0}, Best loss = {1}'.format(best_state, best_loss))
    return best_state, best_loss, best_loss_stats


def shortcut_conns(numlayers, start_everys=[np.inf, 4, 2], loss_kw={}):
    '''
    shortcuts can be none, half or full
    One of them is already done, do the rest here
    '''
    states = []
    loss_stats = []
    losses = np.asarray([])

    for start_every in start_everys:
        if start_every == form_shortcuts_start_every(numlayers):
            continue  # don't repeat the shortcut config already done
        states.append({'shortcuts': form_shortcuts(numlayers, start_every=start_every)})
        loss_stats.append(lossfunc(state=states[-1], **loss_kw))
        losses = np.append(losses, loss_stats[-1]['loss'])
        printf('State = {0}, Loss = {1}\n'.format(states[-1], losses[-1]))

    best_pos, best_loss = np.argmin(losses), np.min(losses)
    best_state, best_loss_stats = states[best_pos], loss_stats[best_pos]
    printf('\nBest state = {0}, Best loss = {1}'.format(best_state, best_loss))
    return best_state, best_loss, best_loss_stats


def dropout_mlp(num_hidden_layers, drop_probs=[0, 0.1, 0.2, 0.3, 0.4, 0.5], loss_kw={}):
    states = []
    loss_stats = []
    losses = np.asarray([])

    for drop_prob in drop_probs:
        if drop_prob == net_kws_defaults['dropout_probs_mlp'][0]:
            continue
        if drop_prob == 0:
            states.append({'apply_dropouts_mlp': num_hidden_layers * [0], 'dropout_probs_mlp': []})
        else:
            states.append({'apply_dropouts_mlp': num_hidden_layers * [1], 'dropout_probs_mlp': num_hidden_layers * [drop_prob]})

    for state in states:
        loss_stats.append(lossfunc(state=state, **loss_kw))
        losses = np.append(losses, loss_stats[-1]['loss'])
        printf('State = {0}, Loss = {1}\n'.format(state, losses[-1]))

    best_pos, best_loss = np.argmin(losses), np.min(losses)
    best_state, best_loss_stats = states[best_pos], loss_stats[best_pos]
    printf('\nBest state = {0}, Best loss = {1}'.format(best_state, best_loss))
    return best_state, best_loss, best_loss_stats


# =============================================================================
# EXECUTION
# =============================================================================
