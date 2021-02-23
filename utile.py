import time


name = int(time.time())
log_file = f"log_{name}.txt"
model_name = f"model_{name}.pt"


def printf(str1, str2=None, str3=None, str4=None):
    logging(str1, log_file)
    if str2 is not None:
        logging(str2, log_file)
        if str3 is not None:
            logging(str3, log_file)
            if str4 is not None:
                logging(str4, log_file)
    return str1[:-1]


def logging(log, file_name):
    print(log)
    with open(file_name, 'a') as f:
        f.write(log + '\n')


def dict_str(dict):
    str1 = ''
    keys = ['val_split',
            'wc',
            'penalize',
            'tbar_epoch',
            'numepochs',
            'val_patience',
            'bo_prior_states',
            'bo_steps',
            'bo_explore',
            'grid_search_order',
            'num_conv_layers',
            'channels_first',
            'embedding_dim',
            'channels_upper',
            'num_hidden_layers',
            'weight_decay',
            'lr',
            'batch_size',
            'input_drop_probs',
            'drop_probs_cnn',
            'drop_probs_mlp',
            'seed',
            'grid_search_order',
            'output_size']
    for k in keys:
        str1 += f'{k} : {dict[k]}'
        str1 += '\n'
    return str1[:-1]
