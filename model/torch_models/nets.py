import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


net_kws_defaults = {
    'act': 'relu',
    'out_channels': [],
    'embedding_dim': [],
    'kernel_sizes': [3],
    'paddings': [],  # Fixed acc to kernel_size, i.e. 1 for k=3, 2 for k=5, etc
    'dilations': [1],
    'groups': [1],
    'strides': [1],
    'apply_maxpools': [0],
    'apply_gap': 1,
    'apply_bns': [1],
    'apply_dropouts': [1],
    'dropout_probs': [0.1, 0.3],  # input layer, other layers
    'shortcuts': [0],
    'hidden_mlp': [],
    'apply_dropouts_mlp': [1],
    'dropout_probs_mlp': [0.2]}


nn_activations = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}

F_activations = {
    'relu': F.relu,
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
}


class Base_Net(nn.Module):
    def __init__(self, input_size=[3, 32, 32], output_size=10, **kw):
        '''
        *** Create Pytorch net ***
        input_size: Iterable. Size of 1 input. Example: [3,32,32] for CIFAR, [784] for MNIST
        output_size: Integer. #labels. Example: 100 for CIFAR-100, 39 for TIMIT phonemes
        kw:
            act: String. Activation for all layers. Must be pre-defined in F_activations and nn_activations. Default 'relu'
            --- CONV ---:
                    out_channels: Iterable. #filters in each conv layer, i.e. #conv layers. If no conv layer is needed, enter []
                --- For the next kws, either pass an iterable of size = size of out_channels, OR leave blank to get default values ---
                        kernel_sizes: Default all 3
                        strides: Default all 1
                        paddings: Default values keep output size same as input for that kernel_size. Example 2 for kernel_size=5, 1 for kernel_size=3
                        dilations: Default all 1
                        groups: Default all 1
                        apply_bns: 1 to get BN layer after the current conv layer, else 0. Default all 1
                        apply_maxpools: 1 to get maxpool layer after the current conv layer, else 0. Default all 0
                        apply_dropouts: 1 to get dropout layer after the current conv layer, else 0. Default all 1
                        shortcuts: 1 to start shortcut after current conv layer, else 0. All shortcuts rejoin after 2 layers. Default all 0
                            2 consecutive elements of shortcuts cannot be 1, last 2 elements of shortcuts must be 0s
                            The shortcut portion has added 0s to compensate for channel increase, and avg pools to compensate for dwensampling
                    dropout_probs: Iterable of size = #1s in apply_dropouts. DROP probabilities for each dropout layer. Default first layer 0.1, all other 0.3
                        Eg: If apply_dropouts = [1,0,1,0], then dropout_probs = [0.1,0.3]. If apply_dropouts = [0,1,1,1], then dropout_probs = [0.3,0.3,0.3]
                    apply_gap: 1 to apply global average pooling just before MLPs, else 0. Default 1
            --- MLP ---:
                    hidden_mlp: Iterable. #nodes in the hidden layers only.
                    apply_dropouts_mlp: Whether to apply dropout after current hidden layer. Iterable of size = number of hidden layers. Default all 0
                    dropout_probs_mlp: As in dropout_probs for conv. Default all 0.5

                    Examples:
                        If input_size=800, output_size=10, and hidden_mlp is not given, or is [], then the config will be [800,10]. By default, apply_dropouts_mlp = [], dropout_probs_mlp = []
                        If input_size=800, output_size=10, and hidden_mlp is [100,100], then the config will be [800,100,100,10]. apply_dropouts_mlp for example can be [1,0], then dropout_probs_mlp = [0.5] by default
            --- NLP ---:
                    embedding_dim = vectors dimension after the embedding layer

        '''
        super().__init__()
        self.act = kw['act'] if 'act' in kw else net_kws_defaults['act']

        # ### Conv ####
        self.out_channels = kw['out_channels'] if 'out_channels' in kw else net_kws_defaults['out_channels']
        self.num_layers_conv = len(self.out_channels)
        self.kernel_sizes = kw['kernel_sizes'] if 'kernel_sizes' in kw else self.num_layers_conv * net_kws_defaults['kernel_sizes']
        self.strides = kw['strides'] if 'strides' in kw else self.num_layers_conv * net_kws_defaults['strides']
        self.paddings = kw['paddings'] if 'paddings' in kw else [(ks - 1) // 2 for ks in self.kernel_sizes]
        self.dilations = kw['dilations'] if 'dilations' in kw else self.num_layers_conv * net_kws_defaults['dilations']
        self.groups = kw['groups'] if 'groups' in kw else self.num_layers_conv * net_kws_defaults['groups']
        self.apply_bns = kw['apply_bns'] if 'apply_bns' in kw else self.num_layers_conv * net_kws_defaults['apply_bns']
        self.apply_maxpools = kw['apply_maxpools'] if 'apply_maxpools' in kw else self.num_layers_conv * net_kws_defaults['apply_maxpools']
        self.apply_gap = kw['apply_gap'] if 'apply_gap' in kw else net_kws_defaults['apply_gap']
        self.shortcuts = kw['shortcuts'] if 'shortcuts' in kw else self.num_layers_conv * net_kws_defaults['shortcuts']
        self.apply_dropouts = kw['apply_dropouts'] if 'apply_dropouts' in kw else (self.num_layers_conv + 1) * net_kws_defaults['apply_dropouts']
        self.conv = nn.ModuleDict({})

        if 'dropout_probs' in kw:
            self.dropout_probs = kw['dropout_probs']
        else:
            self.dropout_probs = np.count_nonzero(self.apply_dropouts) * [net_kws_defaults['dropout_probs'][1]]
            if len(self.apply_dropouts) != 0 and self.apply_dropouts[0] == 1:
                self.dropout_probs[0] = net_kws_defaults['dropout_probs'][0]

        # ### MLP ####
        self.mlp_input_size = self.get_mlp_input_size(input_size, self.conv)
        self.n_mlp = [self.mlp_input_size, output_size]
        if 'hidden_mlp' in kw:
            self.n_mlp[1:1] = kw['hidden_mlp']  # now n_mlp has the full MLP config, e.g. [800,100,10]
        self.num_hidden_layers_mlp = len(self.n_mlp[1:-1])
        self.apply_dropouts_mlp = kw['apply_dropouts_mlp'] if 'apply_dropouts_mlp' in kw else self.num_hidden_layers_mlp * net_kws_defaults['apply_dropouts_mlp']
        self.dropout_probs_mlp = kw['dropout_probs_mlp'] if 'dropout_probs_mlp' in kw else np.count_nonzero(self.apply_dropouts_mlp) * net_kws_defaults['dropout_probs_mlp']

        self.mlp = nn.ModuleList([])
        for i in range(len(self.n_mlp) - 1):
            self.mlp.append(nn.Linear(self.n_mlp[i], self.n_mlp[i + 1]))
        # # Do NOT put dropouts here instead, use F.dropout in forward()

        self.output_size = output_size
        # ### NLP ###
        self.is_nlp = bool('embedding_dim' in kw)
        if self.is_nlp:
            self.embedding_dim = kw['embedding_dim']
            self.embedding = nn.Embedding(input_size[0], self.embedding_dim, padding_idx=0)
            self.linear_out = nn.Linear(self.out_channels[-1], self.output_size)
            self.apply_gap = kw['apply_gap'] if 'apply_gap' in kw else [0]
            self.apply_bns = kw['apply_bns'] if 'apply_bns' in kw else self.num_layers_conv * [0]
            self.apply_dropouts_mlp = kw['apply_dropouts_mlp'] if 'apply_dropouts_mlp' in kw else net_kws_defaults['apply_dropouts_mlp']
            self.dropout_probs_mlp = kw['dropout_probs_mlp'] if 'dropout_probs_mlp' in kw else net_kws_defaults['dropout_probs_mlp']
            self.apply_dropouts_embedding = self.apply_dropouts[0]
            self.apply_dropouts = self.apply_dropouts[1:]
            if self.apply_dropouts_embedding:
                self.embedding_dropout_prob = self.dropout_probs[0]
                self.embedding_dropout_layer = nn.Dropout(self.embedding_dropout_prob)
                self.dropout_probs = self.dropout_probs[1:]
            self.flat = nn.Flatten()

    def init_weight(self):
        if self.is_nlp:
            nn.init.normal_(self.linear_out.weight, std=0.01)
            nn.init.zeros_(self.linear_out.bias)
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.embedding.weight[0, :])
            for layer in self.conv:
                if isinstance(self.conv[layer], nn.modules.conv.Conv1d):
                    torch.nn.init.normal_(self.conv[layer].weight, mean=0, std=0.01)

    def get_mlp_input_size(self, input_size, prelayers):
        x = torch.ones(1, *input_size)  # dummy input: all 1s, batch size 1
        with torch.no_grad():
            for layer in prelayers:
                x = prelayers[layer](x)
        return np.prod(x.size()[1:])


class Search_Net(Base_Net):
    def __init__(self, input_size=[3, 32, 32], output_size=10, **kw):
        super(Search_Net, self).__init__(input_size, output_size, **kw)
        dropout_index = 0
        if self.is_nlp:
            for i in range(self.num_layers_conv):
                if i > 0:
                    self.conv['act-{0}'.format(i)] = nn_activations[self.act]()
                self.conv['conv-{0}'.format(i)] = nn.Conv1d(
                    in_channels=self.embedding_dim if i == 0 else self.out_channels[i - 1],
                    out_channels=self.out_channels[i],
                    kernel_size=self.kernel_sizes[i],
                    stride=1,
                    padding=1,
                    bias=False
                )

                if self.apply_maxpools[i] == 1:
                    self.conv['mp-{0}'.format(i)] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

                if self.apply_bns[i] == 1:
                    self.conv['bn-{0}'.format(i)] = nn.BatchNorm2d(self.out_channels[i])
                if i == 0:
                    # self.conv['act-{0}'.format(i)] = nn_activations[self.act]()
                    pass

                if self.apply_dropouts[i] == 1:
                    self.conv['drop-{0}'.format(i)] = nn.Dropout(self.dropout_probs[dropout_index])
                    dropout_index += 1
        else:
            for i in range(self.num_layers_conv):
                self.conv['conv-{0}'.format(i)] = nn.Conv2d(
                    in_channels=input_size[0] if i == 0 else self.out_channels[i - 1],
                    out_channels=self.out_channels[i],
                    kernel_size=self.kernel_sizes[i],
                    stride=self.strides[i],
                    padding=self.paddings[i],
                    dilation=self.dilations[i],
                    groups=self.groups[i]
                )

                if self.apply_maxpools[i] == 1:
                    self.conv['mp-{0}'.format(i)] = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

                if self.apply_bns[i] == 1:
                    self.conv['bn-{0}'.format(i)] = nn.BatchNorm2d(self.out_channels[i])

                self.conv['act-{0}'.format(i)] = nn_activations[self.act]()

                if self.apply_dropouts[i] == 1:
                    self.conv['drop-{0}'.format(i)] = nn.Dropout(self.dropout_probs[dropout_index])
                    dropout_index += 1
        if self.apply_gap == 1 and self.num_layers_conv > 0:  # GAP is not done when there are no conv layers
            self.conv['gap'] = nn.AdaptiveAvgPool2d(output_size=1)  # this is basically global average pooling, i.e. input of (batch,cin,h,w) is converted to output (batch,cin,1,1)

        self.init_weight()

    def forward(self, x):
        if self.is_nlp:
            x = self.nlp_forward(x)
        else:
            x = self.normal_forward(x)
        return x

    def nlp_forward(self, x):
        rejoin = -1
        if not x.dtype == torch.long:
            x = torch.randint(1, 150, size=x.shape)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        if self.apply_dropouts_embedding:
            x = self.embedding_dropout_layer(x)
        for layer in self.conv:
            if isinstance(self.conv[layer], nn.modules.conv.Conv1d):
                block = int(layer.split('-')[1])
                if block > 0 and self.shortcuts[block - 1] == 1:
                    rejoin = block + 1
                    y = x.clone()
                    count_downsampling = sum(self.apply_maxpools[block:block + 2]) + sum(self.strides[block:block + 2]) - 2
                    for _ in range(count_downsampling):
                        y = F.avg_pool1d(y, kernel_size=2, ceil_mode=True)
                    y = F.pad(y, (0, 0, self.out_channels[block + 1] - self.out_channels[block - 1], 0))
            if block == rejoin and 'act' in layer:  # add shortcut to residual just before activation
                x += y

            x = self.conv[layer](x)
        x = x.max(2, keepdim=True)[0]
        x = self.flat(x)
        if self.apply_dropouts_mlp[0] == 1:
            x = F.dropout(x, p=self.dropout_probs_mlp[0])
        x = self.linear_out(x)
        return x

    def normal_forward(self, x):
        rejoin = -1
        for layer in self.conv:
            if isinstance(self.conv[layer], nn.modules.conv.Conv2d):
                block = int(layer.split('-')[1])
                if block > 0 and self.shortcuts[block - 1] == 1:
                    rejoin = block + 1
                    y = x.clone()
                    count_downsampling = sum(self.apply_maxpools[block:block + 2]) + sum(self.strides[block:block + 2]) - 2
                    for _ in range(count_downsampling):
                        y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
                    y = F.pad(y, (0, 0, 0, 0, 0, self.out_channels[block + 1] - self.out_channels[block - 1], 0, 0))
            if block == rejoin and 'act' in layer:  # add shortcut to residual just before activation
                x += y

            x = self.conv[layer](x)

        x = x.view(-1, self.mlp_input_size)  # flatten data to MLP inputs
        dropout_index = 0
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i != len(self.mlp) - 1:  # last layer should not have regular activation
                x = F_activations[self.act](x)
                if self.apply_dropouts_mlp[i] == 1:
                    x = F.dropout(x, p=self.dropout_probs_mlp[dropout_index])
                    dropout_index += 1
        return x


Net = Search_Net
