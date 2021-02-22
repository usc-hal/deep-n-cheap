import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import numpy as np
import os

PAD_IDX = 0
OOV_TOK = '!OOV!'
Tok_ext = '.tok.txt'
Cat_ext = '.cat'
Vocab_ext = '.vocab.txt'


def get_saved_data(name, batch_size, data_folder, val_split, seed):
    if name == 'yelp.2':
        data_size = 560_000
        dataset = 'yelppol'
        file_name = 'data_pol'
    elif name == 'yelp.5':
        data_size = 650_000
        dataset = 'yelpful'
        file_name = "data_full"

    val_size = int(val_split * data_size)
    train_size = data_size - val_size
    cfg = {
        'dataroot': data_folder + file_name,
        'dataset': dataset,
        'num_train': train_size,
        'num_dev': val_size,
        'batch_size': batch_size,
        'batch_unit': 32,
        'req_max_len': -1,
        'seed': seed}
    # data = prepare_yelp_big(batch_size, 'YelpReviewPolarity')
    # data = prepare_yelp_big(batch_size, 'YelpReviewFull')
    data = DPCNN_data(cfg)
    return data


def prepare_yelp_big(batch_size, name, data_path='/home/babakniy/'):
    # import os
    # NGRAMS = 1
    # dic_len = 30_000
    # if not os.path.isdir('./.data'):
    #     os.mkdir('./.data')
    # if name == 'YelpReviewPolarity':
    #     voc_path = '/home/babakniy/polar.pth'
    # else:
    #     voc_path = '/home/babakniy/full_vocab.pth'
    # new_counter = torch.load(voc_path)
    # new_counter[''] = 0
    # new_counter = dict(sorted(new_counter.items(), key=lambda item: item[1], reverse=True)[:dic_len])
    # specials = ('<unk>', '<pad>')
    # for s in specials:
    #     new_counter[s] = 0

    # import torchtext
    # from torchtext.datasets import text_classification
    # my_vocab = torchtext.vocab.Vocab(new_counter)
    # train_dataset, test_dataset = text_classification.DATASETS[name](root='./.data', ngrams=NGRAMS, vocab=my_vocab)
    # torch.save(train_dataset, name + '_train.pt')
    # torch.save(test_dataset, name + '_test.pt')
    train_dataset = torch.load(data_path + name + '_train.pt')
    test_dataset = torch.load(data_path + name + '_test.pt')
    pad_id = train_dataset.get_vocab()['<pad>']
    train_len = int(len(train_dataset) * 0.9)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    def batching(batch):
        max_len = 400
        label = torch.tensor([entry[0] for entry in batch])
        text = [entry[1][: max_len] for entry in batch]
        textLength = torch.tensor([entry.shape for entry in text])
        max_text_length = torch.max(textLength)
        newtext = [torch.cat((text[i], torch.zeros([max_text_length - textLength[i]], dtype=torch.long) + pad_id), 0)
                   for i in range(len(text))]
        text2 = torch.stack(newtext)  # the padded tensor
        return text2, label

    train_loader = DataLoader(sub_train_, batch_size=batch_size, shuffle=True, collate_fn=batching)
    valid_loader = DataLoader(sub_valid_, batch_size=batch_size, shuffle=False, collate_fn=batching)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=batching)
    data = {
        'type': 'loader',
        'train': train_loader,
        'val': valid_loader,
        'test': test_loader}
    return data


def DPCNN_data(cfg):
    trn_dlist, dev_dlist = get_dlist(cfg['seed'], cfg['num_train'], cfg['num_dev'], None, None, len(prep_uni('train', cfg['dataroot'], cfg['dataset'])))
    td_ds, td_ls = read_ds_ls_x(trn_dlist, cfg['dataroot'], cfg['dataset'])  # training data
    dv_ds, dv_ls = read_ds_ls_x(dev_dlist, cfg['dataroot'], cfg['dataset'])  # validation data
    type = 'test'
    ts_ds = prep_uni(type, cfg['dataroot'], cfg['dataset'])
    ts_ls = prep_lab(type, cfg['dataroot'], cfg['dataset'])
    bch_param = {'req_max_len': cfg['req_max_len'], 'batch_unit': cfg['batch_unit'], 'batch_size': cfg['batch_size']}

    trn_data = TextDataBatches(td_ds, td_ls, **bch_param, do_shuffle=True)
    dev_data = TextDataBatches(dv_ds, dv_ls, **bch_param, do_shuffle=False)
    tst_data = TextDataBatches(ts_ds, ts_ls, **bch_param, do_shuffle=False)
    data = {
        'type': 'loader',
        'train': trn_data,
        'val': dev_data,
        'test': tst_data}
    return data


class TextData_Uni():
    def __init__(self, pathname=None, min_dlen=-1, num_max=-1, d_num_max=-1):
        self._ids = []
        self._ids_top = [0]
        self._vocab = None
        self._d_list = []
        if pathname is not None and pathname != '':
            self.load(pathname)
            min_dlen = max(0, min_dlen)
            self._ids, self._ids_top, self._d_list = remove_short_ones(self._ids, self._ids_top, min_dlen,
                                                                       do_d_list=True)
            if num_max > 0 or d_num_max > 0:
                self._ids, self._ids_top, self._d_list = remove_randomly(self._ids, self._ids_top, d_list=self._d_list,
                                                                         num_max=num_max, d_num_max=d_num_max)
            self._d_list = tuple(self._d_list)

    # ---  Reorder documents according to dxs (a list of document#'s).
    def shuffle(self, dxs):
        self._ids, self._ids_top = shuffle_ids(dxs, self._ids, self._ids_top)
        self._ids = tuple(self._ids)
        self._ids_top = tuple(self._ids_top)
        self.d_list = []

    def is_same(self, oth):
        return are_these_same(self, oth, ['_ids', '_ids_top', '_vocab', '_d_list'])

    def d_list(self):
        return self._d_list

    def num_datasets(self):
        return 1

    def get_ids(self):
        return self._ids, self._ids_top

    def vocab(self):
        return self._vocab

    def n_min(self):
        return 1

    def n_max(self):
        return 1

    def rev_vocab(self):
        return gen_rev_vocab(self._vocab)

    def save(self, pathname):
        torch.save(dict(ids=self._ids, ids_top=self._ids_top, vocab=self._vocab), pathname)

    def load(self, pathname):
        d = torch.load(pathname)
        self._ids = tuple(d['ids'])
        self._ids_top = tuple(d['ids_top'])
        self._vocab = d['vocab']
        check_ids(self._ids, self.vocab_size())

    def create(self, tok_pathname, vocab_pathname, do_oov):
        self._vocab = read_vocab(vocab_pathname, do_oov=do_oov)
        self._ids, self._ids_top = tokens_to_ids(tok_pathname, self._vocab, do_oov=do_oov)

    def __len__(self):
        return len(self._ids_top) - 1

    def vocab_size(self):
        if self._vocab is None:
            return 0
        else:
            return max(self._vocab.values()) + 1

    def data_len(self, d):
        self._check_index(d)
        return (self._ids_top[d + 1] - self._ids_top[d])

    def _check_index(self, d):
        if d < 0 or d >= len(self):
            raise IndexError

    def __getitem__(self, d):
        self._check_index(d)
        return self._ids[self._ids_top[d]:self._ids_top[d + 1]]


class TextData_Lab(object):
    def __init__(self, pathname=None):
        self._labels = []
        self._num_class = 0
        if pathname is not None and pathname != '':
            self.load(pathname)

    # ---  Reorder lables according to the order of dxs (a list of document#'s).
    def shuffle(self, dxs):
        new_labels = []
        for d in dxs:
            self._check_index(d)
            new_labels += [self._labels[d]]

        self._labels = tuple(new_labels)

    # ---
    def save(self, pathname):
        torch.save(dict(labels=self._labels, num_class=self._num_class), pathname)

    def load(self, pathname):
        d = torch.load(pathname)
        self._labels = tuple(d['labels'])
        self._num_class = d['num_class']

    def create(self, lab_pathname, catdic_pathname):
        self._labels, self._num_class = read_labels(lab_pathname, catdic_pathname)

    def __len__(self):
        return len(self._labels)

    def num_class(self):
        return self._num_class

    def _check_index(self, d):
        if d < 0 or d >= len(self):
            raise IndexError

    def __getitem__(self, d):
        self._check_index(d)
        return self._labels[d]


class TextDataBatches(object):
    def __init__(self, dataset, labset, batch_size, do_shuffle, req_max_len,
                 batch_unit=-1, req_min_len=-1, x_dss=None):
        assert dataset is not None
        assert labset is not None
        if len(dataset) != len(labset):
            raise Exception('Conflict in sizes of dataset and labset')

        self.ds = dataset   # TextData_Uni
        self.ls = labset    # TextData_Lab
        self.x_dss = x_dss  # list of TextData_N

        self._batches = []  # list of list, e.g., [[10,3,0], [2,5,2], ... ]

        self._batch_unit = batch_unit
        self._batch_size = batch_size
        self._do_shuffle = do_shuffle

        self._dxs = []
        self._dxs_pos = 0
        self._req_min_len = req_min_len
        self._req_max_len = req_max_len

        self._create_dxs()

        if batch_unit > 0:
            self._make_batches()

    def num_batches(self):
        if self._batch_unit > 0:
            return len(self._bxs)
        else:
            return max(1, len(self._dxs) // self.batch_size)

    def _create_dxs(self):
        self._dxs = list(range(len(self.ds)))
        self._dxs_pos = 0

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        return self

    def __next__(self):
        if self._batch_unit > 0:  # text lengths in a batch are roughly the same.  fast.
            if self._bxs_pos >= len(self._bxs):
                self._bxs_pos = 0  # prepare for the next epoch
                raise StopIteration

            if self._bxs_pos == 0 and self._do_shuffle:
                self._make_batches()

            bx = self._bxs[self._bxs_pos]  # batch id
            idx = self._batches[bx]
            self._bxs_pos += 1

        else:  # text lengths in a batch vary a lot.  good randomization but slow.
            num = min(self._batch_size, len(self._dxs) - self._dxs_pos)
            if num <= 0:
                self._dxs_pos = 0  # prepare for the next epoch
                raise StopIteration

            if self._dxs_pos == 0 and self._do_shuffle:
                np.random.shuffle(self._dxs)

            idx = [self._dxs[self._dxs_pos + i] for i in range(num)]
            self._dxs_pos += self._batch_size

        if self.x_dss is None:
            return self._gen_batch(idx)
        else:
            return self._gen_batch_x(idx)

    def _gen_batch(self, idx):  # return [data, labels]
        max_len = self._req_min_len
        for d in idx:
            max_len = max(max_len, self.ds.data_len(d))

        # padding
        data = [(list(self.ds[d]) + [PAD_IDX for i in range(max_len - self.ds.data_len(d))]) for d in idx]

        # shortening
        if self._req_max_len > 0 and self._req_max_len < max_len:
            data = [data[i][0:self._req_max_len] for i in range(len(data))]

        data = cast(torch.tensor(data), 'long')
        labels = [self.ls[d] for d in idx]
        labels = cast(torch.tensor(labels), 'long')
        return [data, labels]

    def _gen_batch_x(self, idx):  # return [data, labels, extra data]
        if len(self.x_dss) > 1:
            raise ValueError('no support (yet) for len(x_dss) > 1')

        data_lab = self._gen_batch(idx)  # [data,labels]
        data_len = data_lab[0].size(1)   # length of data

        x_ds = self.x_dss[0]
        # padding
        x_data = [(list(x_ds[d]) + [PAD_IDX for i in range(data_len - x_ds.data_len(d))]) for d in idx]
        # shortening
        x_data = [x_data[i][0:data_len] for i in range(len(x_data))]
        x_data = cast(torch.tensor(x_data), 'long')

        return data_lab + [[x_data]]

    #  Sort documents roughly by length.  "unit" is for making it "rough".
    def _make_batches(self):
        unit = self._batch_unit
        dict = {}
        dxs = [i for i in range(len(self))]
        if self._do_shuffle:
            np.random.shuffle(dxs)
        total_len = 0
        for i in range(len(dxs)):
            dx = dxs[i]
            data_len = self.ds.data_len(dx)
            dict[dx] = data_len // unit
            total_len += data_len

        sorted_d = sorted(dict.items(), key=lambda x: x[1])  # Roughly sort by length
        self._dxs = []
        for i in range(len(sorted_d)):
            self._dxs += [sorted_d[i][0]]

        num_batches = (len(self._dxs) + self._batch_size - 1) // self._batch_size

        # -----
        self._batches = []
        sz = len(self)

        if self._batch_size > sz:
            raise Exception('batch_size must not exceed the size of dataset: ' + str(self._batch_size) + ',' + str(sz))

        for b in range(num_batches):
            dxs_pos = b * self._batch_size
            num = min(self._batch_size, sz - dxs_pos)
            self._batches += [[self._dxs[dxs_pos + i] for i in range(num)]]

        # -----
        self._bxs = [i for i in range(num_batches)]
        if self._do_shuffle:
            np.random.shuffle(self._bxs)

        self._bxs_pos = 0


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def prep_uni(type, dataroot, dataset):
    return TextData_Uni(pathname=gen_uni_name(dataroot, dataset, type))


def tokens_to_ids(pathname, vocab, do_oov=False):
    with open(pathname, encoding="utf8") as f:
        docs = f.read().split('\n')

    num_docs = len(docs)
    if docs[-1] == '':
        num_docs -= 1

    ids_top = [0 for i in range(num_docs + 1)]
    num_tokens = 0
    for d in range(num_docs):
        num_tokens += len(docs[d].split())
        ids_top[d + 1] = num_tokens

    oov_id = 0
    if do_oov:
        oov_id = vocab.get(OOV_TOK)
        if oov_id is None:
            raise Exception('oov_id is None??')

    oov = 0
    ids = [0 for i in range(num_tokens)]
    max_len = 0
    for d in range(num_docs):

        tokens = docs[d].lower().split()
        max_len = max(max_len, len(tokens))

        for i in range(len(tokens)):
            index = vocab.get(tokens[i])
            if index is not None:
                ids[ids_top[d] + i] = index
            else:
                if do_oov:
                    ids[ids_top[d] + i] = oov_id
            oov += 1
    return ids, ids_top


def get_dlist(seed, num_train, num_dev, train_dlist_path, dev_dlist_path, num_data):
    if num_dev > 0:
        if num_train <= 0:
            num_train = max(1, num_data - num_dev)

        rs = np.random.get_state()
        np.random.seed(seed)
        if num_train + num_dev > num_data:
            raise ValueError('num_train + num_dev must be no greater than %d.' % num_data)

        indexes = [d for d in range(num_data)]
        np.random.shuffle(indexes)
        trn_dlist = indexes[0:num_train]
        dev_dlist = indexes[num_train: num_train + num_dev]
        np.random.set_state(rs)

    elif train_dlist_path and dev_dlist_path:
        def read_dlist(path):
            with open(path, encoding="utf8") as f:
                input = f.read().split('\n')
            dlist = [int(inp) for inp in input if len(inp) > 0]
            return dlist

        trn_dlist = read_dlist(train_dlist_path)
        dev_dlist = read_dlist(dev_dlist_path)
    else:
        raise ValueError('Either num_dev/num_train or train_dlist_path/dev_dlist_path is needed.')

    return trn_dlist, dev_dlist


def read_ds_ls_x(dlist, dataroot, dataset):  # read data and labels
    type = 'train'
    ds = prep_uni(type, dataroot, dataset)
    ds.shuffle(dlist)
    ls = prep_lab(type, dataroot, dataset)
    ls.shuffle(dlist)
    return ds, ls


def prep_lab(type, dataroot, dataset):
    return TextData_Lab(pathname=gen_lab_name(dataroot, dataset, type))


def gen_lab_name(dataroot, dataset, type):
    return where(dataroot) + dataset + '-' + type + '-lab.pth'


def read_labels(lab_pathname, catdic_pathname):
    with open(lab_pathname) as f:
        labels = f.read().split()
    with open(catdic_pathname) as f:
        list_cat = f.read().split()
        dic_cat = {list_cat[i]: i for i in range(len(list_cat))}

    return [dic_cat[labels[i]] for i in range(len(labels))], len(list_cat)


def read_vocab(pathname, do_oov=False):
    with open(pathname, encoding="utf8") as f:
        input = f.read().split('\n')
    vocab_size = len(input)
    if input[-1] == '':
        vocab_size -= 1

    vocab = {input[i].strip().split()[0]: i + 1 for i in range(vocab_size)}  # '+1' for reserving 0 for padding
    vocab[''] = 0

    if do_oov:
        if OOV_TOK in vocab.keys():
            raise Exception('OOV_TOK exists in text?!')
        vocab[OOV_TOK] = max(vocab.values()) + 1

    return vocab


def remove_short_ones(ids, ids_top, min_dlen, do_d_list=False):
    d_num = len(ids_top) - 1
    new_ids = []
    new_ids_top = []
    d_list = [] if do_d_list else None
    for d in range(d_num):
        dlen = ids_top[d + 1] - ids_top[d]
        if dlen < min_dlen:
            continue

        new_ids_top += [len(new_ids)]
        new_ids += ids[ids_top[d]:ids_top[d + 1]]
        if d_list is not None:
            d_list += [d]

    new_ids_top += [len(new_ids)]
    return tuple(new_ids), tuple(new_ids_top), d_list


def remove_randomly(ids, ids_top, num_max, d_list=None, d_num_max=-1):
    org_d_num = len(ids_top) - 1
    org_num = len(ids)

    if (d_num_max <= 0 or org_d_num <= d_num_max) and (num_max <= 0 or org_num <= num_max):
        return ids, ids_top, d_list

    dxs = torch.randperm(org_d_num).tolist()
    new_ids_top = []
    new_ids = []
    new_d_list = None if d_list is None else []
    for d in dxs:
        new_ids_top += [len(new_ids)]
        new_ids += ids[ids_top[d]:ids_top[d + 1]]
        if new_d_list is not None:
            new_d_list += [d_list[d]]

        d_num = len(new_ids_top) - 1
        num = len(new_ids)

        if (d_num_max > 0 and d_num >= d_num_max):
            break
        if (num_max > 0 and num >= num_max):
            break

    new_ids_top += [len(new_ids)]

    return tuple(new_ids), tuple(new_ids_top), new_d_list


def check_ids(ids, vocab_size):
    if min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError('id is out of range: min=%d, max=%d, vocab_size=%d' % (min(ids), max(ids), vocab_size))


def are_these_same(o0, o1, names):
    for name in names:
        if getattr(o0, name) != getattr(o1, name):
            return False
    return True


def gen_uni_name(dataroot, dataset, type):
    return where(dataroot) + dataset + '-' + type + '-uni.pth'


def where(dataroot):
    if dataroot:
        return dataroot + os.path.sep
    else:
        return ''


def gen_rev_vocab(vocab):
    max_value = max(vocab.values())
    v_to_k = ['' for i in range(0, max_value + 1)]
    for k, v in vocab.items():
        v_to_k[v] = k
    return v_to_k


def shuffle_ids(dxs, ids, ids_top):
    d_num = len(ids_top) - 1
    new_ids_top = []
    new_ids = []
    for dx in dxs:
        if dx < 0 or dx >= d_num:
            raise IndexError
        new_ids_top += [len(new_ids)]
        new_ids += ids[ids_top[dx]:ids_top[dx + 1]]
    new_ids_top += [len(new_ids)]
    return new_ids, new_ids_top
