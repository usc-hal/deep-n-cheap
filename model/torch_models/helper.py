import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
import time
import torchnet as tnt
from utile import printf, model_name
from model.torch_models.nets import Net
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


run_kws_defaults = {
    'lr': 1e-3,
    'gamma': 0.2,
    'milestones': [0.25, 0.5, 0.75],
    'weight_decay': 0.,
    'batch_size': 256
}


# =============================================================================
# Classes
# =============================================================================
class Hook():
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, layer, input, output):
        self.output = output

    def close(self):
        self.hook.remove()
# =============================================================================


# =============================================================================
# Helper methods
# =============================================================================
def get_numparams(input_size, output_size, net_kw):
    '''  Get number of parameters in any net '''
    net = Net(input_size=input_size, output_size=output_size, **net_kw)
    numparams = sum([param.nelement() for param in net.parameters()])
    return numparams


def train_batch(x, y, net, lossfunc, opt):
    ''' Train on 1 batch of data '''
    opt.zero_grad()
    out = net(x)
    loss = lossfunc(out, y)
    loss.backward()
    # nn.utils.clip_grad_norm_(net.parameters(), 1)
    opt.step()
    return loss, out


def eval_data(net, x, ensemble, y, **kw):
    '''
    *** General method to run only forward pass using net on data x ***
    kw:
        y : If given, find predictions and compute correct, else correct = None. Obviously y.shape[0] should be equal to x.shape[0]
        lossfunc : If given, compute loss, else loss = None
        hook : If desired, pass as a list of Hook(layer) objects
            E.g: hook = [Hook(layer) for layer in net.conv.layers if 'conv2d' in str(type(layer)).lower()]
            This finds intermediate outputs of all conv2d layers in the net
            These intermediate outputs are returned as raw_layer_outputs, and can be accessed as raw_layer_outputs[i].output
            Else raw_layer_outputs = None
    '''
    net.eval()
    with torch.no_grad():
        out = net(x)
        raw_layer_outputs = kw['hook'] if 'hook' in kw else None
        loss = kw['lossfunc'](out, y).item() if 'lossfunc' in kw else None
    return loss, raw_layer_outputs, out


def save_net(net=None, recs=None, filename='./results_new/new'):
    '''
    *** Saves net and records (if given) ***
    There are 2 ways to save a Pytorch net (see https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        1 - Save complete net: torch.save(net, filename.pt)
            Doing this needs a lot of space, e.g. >200MB for Reuters 100 epochs, hence not practical
            Loading is easy: net = torch.load(filename.pt)
        2 - Save state dict: torch.save(net.state_dict(), filename.pt)
            This needs LOT less space, e.g. 4 MB for Reuters 100 epochs
            Loading needs creation of the Net class with original kw:
                net = Net(args)
                net.load_state_dict(torch.load(filename.pt))
                SO STORE THE NET ARGS ALONG WITH NET, LIKE MAYBE IN A TEXT OR EXCEL FILE
    Use torch.load(map_location='cpu') to load CuDA models on local machine
    This message might appear when doing net.load_state_dict() - IncompatibleKeys(missing_keys=[], unexpected_keys=[]). IGNORE!
    '''
    if recs:
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(recs, f)
    if net:
        torch.save(net.state_dict(), filename + '.pt')


def validate_net(loader, verbose, net, ensemble, epoch, name, lossfunc, val_loader=None, x=None, y=None):
    # mtr_err_val = tnt.meter.ClassErrorMeter(topk=[1], accuracy=False)
    # mtr_loss_val = tnt.meter.AverageValueMeter()
    # if not loader:
    #     loss, _, outputs = eval_data(net=net, x=x, ensemble=ensemble, y=y, lossfunc=lossfunc)
    #     mtr_loss_val.add(float(loss))
    #     mtr_err_val.add(outputs.cpu().detach().numpy(), y)
    # else:
    #     for batch in tqdm(val_loader, leave=False):
    #         inputs, labels = tuple(x.to(device) for x in batch)
    #         loss, _, outputs = eval_data(net=net, x=inputs, ensemble=ensemble, y=labels, lossfunc=lossfunc)
    #         mtr_loss_val.add(float(loss))
    #         mtr_err_val.add(outputs.cpu().detach().numpy(), labels)
    net.eval()
    loss = 0
    acc = 0
    for input, label in val_loader:
        input, label = input.to(device), label.to(device)
        with torch.no_grad():
            output = net(input)
            loss = lossfunc(output, label)
            loss += loss.item()
            acc += (output.argmax(1) == label).sum().item()

    loss = loss.detach().cpu().numpy() / len(val_loader)
    acc = acc / len(val_loader) * 100
    # acc = 100 - (float(mtr_err_val.value()[0]))
    # loss = float(mtr_loss_val.value()[0])
    if verbose:
        acc_r = np.round(acc, 2)
        loss_r = np.round(loss, 3)
        printf(f'{name} Acc = {acc_r}%, Loss = {loss_r}')
    return acc, loss, output
# =============================================================================


# =============================================================================
# Main method to run network
# =============================================================================
def run_network(
    data, input_size, output_size, problem_type, net_kw, run_kw,
    num_workers=8, pin_memory=True,
    validate=True, val_patience=np.inf, test=False, ensemble=False,
    numepochs=100, wt_init=None, bias_init=None, verbose=True
):
    '''
    ARGS:
        data:
            6-ary tuple (xtr,ytr, xva,yva, xte,yte) from get_data_mlp(), OR
            Dict with keys 'train', 'val', 'test' from get_data_cnn()
        input_size, output_size, net_kw : See Net()
        run_kw:
            lr: Initial learning rate
            gamma: Learning rate decay coefficient
            milestones: When to step decay learning rate, e.g. 0.5 will decay lr halfway through training
            weight_decay: Default 0
            batch_size: Default 256
        num_workers, pin_memory: Only required if using Pytorch data loaders
            Generally, set num_workers equal to number of threads (e.g. my Macbook pro has 4 cores x 2 = 8 threads)
        validate: Whether to do validation at the end of every epoch.
        val_patience: If best val acc doesn't increase for this many epochs, then stop training. Set as np.inf to never stop training (until numepochs)
        test: True - Test at end, False - don't
        ensemble: If True, return feedforward soft outputs to be later used for ensembling
        numepochs: Self explanatory
        wt_init, bias_init: Respective pytorch functions
        verbose: Print messages

    RETURNS:
        net: Complete net
        recs: Dictionary with a key for each stat collected and corresponding value for all values of the stat
    '''
# =============================================================================
#     Create net
# =============================================================================
    net = Net(input_size=input_size, output_size=output_size, **net_kw)
    wt_init = wt_init if wt_init is not None else nn.init.kaiming_normal_

    bias_init = bias_init if bias_init is not None else (lambda x: nn.init.constant_(x, 0.1))
    printf(summary(net, input_size=(100,)))
    # Use GPUs if available ##
    net.to(device)  # convert parameters

    # Initialize MLP params ##
    if wt_init is not None:
        wt_init(net.linear_out.weight.data)
    if bias_init is not None:
        bias_init(net.linear_out.bias.data)

# =============================================================================
#     Hyperparameters for the run
# =============================================================================
    lr = run_kw['lr'] if 'lr' in run_kw else run_kws_defaults['lr']
    gamma = run_kw['gamma'] if 'gamma' in run_kw else run_kws_defaults['gamma']  # previously used value according to decay = 1e-5 in keras = 0.9978 for ExponentialLR
    milestones = run_kw['milestones'] if 'milestones' in run_kw else run_kws_defaults['milestones']
    weight_decay = run_kw['weight_decay'] if 'weight_decay' in run_kw else run_kws_defaults['weight_decay']
    batch_size = run_kw['batch_size'] if 'batch_size' in run_kw else run_kws_defaults['batch_size']
    print(lr, gamma, milestones, weight_decay)
    print(40 * "*")
    if not isinstance(batch_size, int):
        batch_size = batch_size.item()  # this is required for pytorch

    if problem_type == 'classification':
        lossfunc = nn.CrossEntropyLoss().to(device)
        # lossfunc = nn.CrossEntropyLoss(reduction='mean')  # IMPORTANT: By default, loss is AVERAGED across samples in a batch. If sum is desired, set reduction='sum'
    elif problem_type == 'regression':
        lossfunc = nn.MSELoss()
    # opt = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=False)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(numepochs * milestone) for milestone in milestones], gamma=gamma)


# =============================================================================
# Data
# =============================================================================
    if type(data) == dict:  # using Pytorch data loaders
        loader = True
        if 'type' in data.keys():
            train_loader, val_loader, test_loader = data['train'], data['val'], data['test']
        else:
            train_loader = torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
            if validate is True:
                val_loader = torch.utils.data.DataLoader(data['val'], batch_size=len(data['val']), num_workers=num_workers, pin_memory=pin_memory)
            if test is True:
                test_loader = torch.utils.data.DataLoader(data['test'], batch_size=len(data['test']), num_workers=num_workers, pin_memory=pin_memory)
    else:  # using numpy
        loader = False
        xtr, ytr, xva, yva, xte, yte = data
# =============================================================================
#     Define records to collect
# =============================================================================
    recs = {
        'train_accs': np.zeros(numepochs), 'train_losses': np.zeros(numepochs),
        'val_accs': np.zeros(numepochs) if validate is True else None, 'val_losses': np.zeros(numepochs) if validate is True else None, 'val_final_outputs': numepochs * [0]  # just initialize a dummy list
        # test_acc and test_loss are defined later
    }
    total_t = 0
    best_val_acc = -np.inf
    best_val_loss = np.inf
    val_model_name = model_name
    topk = [1]
    numbatches = int(np.ceil(xtr.shape[0] / batch_size)) if not loader else len(train_loader)

# =============================================================================
#         Run epoch
# =============================================================================
    # mtr_err = tnt.meter.ClassErrorMeter(topk=topk, accuracy=False)
    # mtr_loss = tnt.meter.AverageValueMeter()
    for epoch in range(numepochs):
        # mtr_err.reset()
        # mtr_loss.reset()
        if verbose:
            printf('Epoch {0}'.format(epoch + 1))

        # Set up epoch ##
        # if not loader:
        #     shuff = torch.randperm(xtr.shape[0])
        #     xtr, ytr = xtr[shuff], ytr[shuff]

        # Train #
        t = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        val_patience_counter = 0
        # for batch in tqdm(range(numbatches) if not loader else train_loader, leave=False):  # Sara
        for batch in train_loader:  # Sara
            # if not loader:
            #     inputs = xtr[batch * batch_size: (batch + 1) * batch_size]  # already converted to device
            #     labels = ytr[batch * batch_size: (batch + 1) * batch_size]
            # else:
            opt.zero_grad()
            input, label = tuple(x.to(device) for x in batch)
            logits = net(input)
            loss = lossfunc(logits, label)
            train_loss += loss.item()
            loss.backward()
            opt.step()
            train_acc += (logits.argmax(1) == label).sum().item()

            # loss, output = train_batch(x=inputs, y=labels, net=net, lossfunc=lossfunc, opt=opt)
            # mtr_loss.add(float(loss))
            # mtr_err.add(output.cpu().detach().numpy(), labels)

        # Time for epoch (don't collect for 1st epoch unless there is only 1 epoch) ##
        t_epoch = time.time() - t
        if epoch > 0 or numepochs == 1:
            total_t += t_epoch

        # Save training records ##
        # recs['train_accs'][epoch] = 100 - (float(mtr_err.value()[0]))
        # recs['train_losses'][epoch] = float(mtr_loss.value()[0])
        recs['train_accs'][epoch] = train_acc / len(train_loader) * 100
        t_loss = train_loss / len(train_loader)
        recs['train_losses'][epoch] = t_loss
        # mtr_err.reset()
        # mtr_loss.reset()
        if verbose:
            printf('Training Acc = {0}%, Loss = {1}'.format(np.round(recs['train_accs'][epoch], 2), np.round(recs['train_losses'][epoch], 3)))  # put \n to make this appear on the next line after progress bar

# =============================================================================
#         Validate (optional)
# =============================================================================
        if validate is True:
            acc, loss, outputs = validate_net(loader, verbose, net, ensemble, epoch, name='Validation', lossfunc=lossfunc,
                                              val_loader=val_loader if loader else None,
                                              x=xva if not loader else None, y=yva if not loader else None)
            recs['val_accs'][epoch] = acc
            recs['val_losses'][epoch] = loss
            recs['val_final_outputs'][epoch] = outputs

# =============================================================================
#       Early stopping logic based on val_acc
# =============================================================================
            if problem_type == 'classification':
                if recs['val_accs'][epoch] > best_val_acc:
                    best_val_acc = recs['val_accs'][epoch]
                    best_val_ep = epoch + 1
                    val_patience_counter = 0  # don't need to define this beforehand since this portion will always execute first when epoch==0
                    torch.save(net.state_dict(), val_model_name)
                else:
                    val_patience_counter += 1
                    if val_patience_counter == val_patience:
                        printf('Early stopped after epoch {0}'.format(epoch + 1))
                        numepochs = epoch + 1  # effective numepochs after early stopping
                        break
            elif problem_type == 'regression':
                if recs['val_losses'][epoch] < best_val_loss:
                    best_val_loss = recs['val_losses'][epoch]
                    best_val_ep = epoch + 1
                    val_patience_counter = 0  # don't need to define this beforehand since this portion will always execute first when epoch==0
                else:
                    val_patience_counter += 1
                    if val_patience_counter == val_patience:
                        printf('Early stopped after epoch {0}'.format(epoch + 1))
                        numepochs = epoch + 1  # effective numepochs after early stopping
                        break
# =============================================================================
#         Schedule hyperparameters
# =============================================================================
        scheduler.step()

# =============================================================================
#     Final stuff at the end of training
# =============================================================================
    # Final val metrics ##
    if validate is True:
        if problem_type == 'classification':
            printf('\nBest validation accuracy = {0}% obtained in epoch {1}'.format(best_val_acc, best_val_ep))
        elif problem_type == 'regression':
            printf('\nBest validation loss = {0} obtained in epoch {1}'.format(best_val_loss, best_val_ep))

    # Testing ##
    if test is True:
        if validate is True:
            net.load_state_dict(torch.load(val_model_name))
        acc, loss, outputs = validate_net(loader, verbose, net, ensemble, epoch, name='Test', lossfunc=lossfunc,
                                          val_loader=test_loader if loader else None,
                                          x=xte if not loader else None, y=yte if not loader else None)

        recs['test_accs'] = acc
        recs['test_losses'] = loss
        # recs['test_final_outputs'] = outputs  @ Sara

    # Avg time taken per epoch ##
    recs['t_epoch'] = total_t / (numepochs - 1) if numepochs > 1 else total_t
    printf('Avg time taken per epoch = {0}'.format(recs['t_epoch']))

    # Cut recs as a result of early stopping ##
    recs = {**{key: recs[key][:numepochs] for key in recs if hasattr(recs[key], '__iter__')}, **{key: recs[key] for key in recs if not hasattr(recs[key], '__iter__')}}  # this cuts the iterables like valaccs to the early stopping point, and keeps single values like testacc unchanged

    return net, recs
# =============================================================================
