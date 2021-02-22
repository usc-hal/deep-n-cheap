import pandas as pd
import numpy as np
# import re
import torch
from torchtext import data
from torchtext.data import Dataset
import dill
from pathlib import Path
import time
import os


def prepare_csv(data_path, save_path):
    df_train = pd.read_csv(data_path + 'train.csv', header=None, names=(['label', 'text']))
    idx = np.arange(df_train.shape[0])
    np.random.shuffle(idx)
    val_size = 5000
    df_train.iloc[idx[val_size:], :].to_csv(save_path + "dataset_train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv(save_path + "dataset_val.csv", index=False)
    df_test = pd.read_csv(data_path + "test.csv", header=None, names=(['label', 'text']))
    df_test.to_csv(save_path + "dataset_test.csv", index=False)


def cleanup_text(texts):
    texts = texts[:300]
    # output = []
    # for text in texts:
    #     sentence = re.sub('[^a-zA-Z]', '', text)  # no punctuations & numbers
    #     sentence = re.sub(r'\s+', '', sentence)  # Removing multiple spaces
    #     sentence = sentence.lower()
    #     if len(sentence) > 2:
    #         output.append(sentence)
    # if len(output) == 0:
    #     output = ["None"]
    return texts


def get_dataset(MAX_VOCAB_SIZE, dir):
    TEXT = data.Field(lower=True, tokenize=lambda s: s.split(), preprocessing=cleanup_text, batch_first=True)
    LABEL = data.LabelField(dtype=torch.long, sequential=False, batch_first=True)
    fields = [('label', LABEL), ('text', TEXT)]
    train, val, test = data.TabularDataset.splits(path=dir, format='csv',
                                                  train='dataset_train.csv',
                                                  validation='dataset_val.csv',
                                                  test="dataset_test.csv",
                                                  fields=fields, skip_header=True)

    TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
    LABEL.build_vocab(train)

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    print(TEXT.vocab.freqs.most_common(20))
    print(TEXT.vocab.itos[:10])
    print(LABEL.vocab.stoi)

    save_dataset(train, "/home/babakniy/DPCNN/train")
    save_dataset(test, "/home/babakniy/DPCNN/test")
    save_dataset(val, "/home/babakniy/DPCNN/val")
    with open("DPCNN/TEXT.Field", "wb")as f:
        dill.dump(TEXT, f)
    with open("DPCNN/LABEL.Field", "wb")as f:
        dill.dump(LABEL, f)


def save_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, path / "examples.pkl", pickle_module=dill)
    torch.save(dataset.fields, path / "fields.pkl", pickle_module=dill)


def load_dataset(path):
    if not isinstance(path, Path):
        path = Path(path)
    examples = torch.load(path / "examples.pkl", pickle_module=dill)
    fields = torch.load(path / "fields.pkl", pickle_module=dill)
    return Dataset(examples, fields)


name = int(time.time())
log_file = f"/home/babakniy/deep-n-cheap-extended/new_log/log_{name}.txt"
model_name = f"/home/babakniy/models/model_{name}.pt"


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
            'is_dpcnn',
            'note',
            'grid_search_order',
            'output_size']
    for k in keys:
        str1 += f'{k} : {dict[k]}'
        str1 += '\n'
    return str1[:-1]


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def printf(str1, str2=None, str3=None, str4=None):
    logging(str1, log_file)
    if str2 is not None:
        logging(str2, log_file)
        if str3 is not None:
            logging(str3, log_file)
            if str4 is not None:
                logging(str4, log_file)
