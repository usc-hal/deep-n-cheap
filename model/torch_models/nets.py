import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


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
    'apply_gap': 0,  # @Sara
    'apply_bns': [0],  # @Sara no bn
    'apply_dropouts': [1],
    'dropout_probs': [0.1, 0.3],  # input layer, other layers
    'shortcuts': [0],
    'hidden_mlp': [],
    'apply_dropouts_mlp': [1],
    'dropout_probs_mlp': [0.2],
}


nn_activations = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    # 'sigmoid': nn.Sigmoid   @ Sara
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
        '''
        super().__init__()
        self.act = kw['act'] if 'act' in kw else net_kws_defaults['act']
        self.out_channels = kw['out_channels'] if 'out_channels' in kw else net_kws_defaults['out_channels']
        self.embedding_dim = kw['embedding_dim'] if 'out_channels' in kw else net_kws_defaults['out_channels']
        self.num_layers_conv = len(self.out_channels)
        self.embedding = nn.Embedding(input_size[0], self.embedding_dim, padding_idx=0)
        self.kernel_sizes = kw['kernel_sizes'] if 'kernel_sizes' in kw else self.num_layers_conv * net_kws_defaults['kernel_sizes']
        self.strides = kw['strides'] if 'strides' in kw else self.num_layers_conv * net_kws_defaults['strides']
        self.paddings = kw['paddings'] if 'paddings' in kw else [(ks - 1) // 2 for ks in self.kernel_sizes]
        self.dilations = kw['dilations'] if 'dilations' in kw else self.num_layers_conv * net_kws_defaults['dilations']
        self.groups = kw['groups'] if 'groups' in kw else self.num_layers_conv * net_kws_defaults['groups']
        self.apply_bns = kw['apply_bns'] if 'apply_bns' in kw else self.num_layers_conv * net_kws_defaults['apply_bns']
        self.apply_maxpools = kw['apply_maxpools'] if 'apply_maxpools' in kw else self.num_layers_conv * net_kws_defaults['apply_maxpools']
        self.apply_gap = kw['apply_gap'] if 'apply_gap' in kw else net_kws_defaults['apply_gap']
        self.shortcuts = kw['shortcuts'] if 'shortcuts' in kw else self.num_layers_conv * net_kws_defaults['shortcuts']
        self.output_size = output_size
        self.flat = nn.Flatten()
        self.linear_out = nn.Linear(self.out_channels[-1], self.output_size)
        self.conv = nn.ModuleDict({})

    def init_weight(self):
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


# class Net_DPCNN(Base_Net):
#     def __init__(self, input_size=[3, 32, 32], output_size=10, **kw):
#         super(Net_DPCNN, self).__init__(input_size, output_size, **kw)
#         for i in range(self.num_layers_conv):
#             self.conv['conv-{0}'.format(i)] = nn.Conv1d(
#                 in_channels=self.embedding_dim if i == 0 else self.out_channels[i - 1],
#                 out_channels=self.out_channels[i],
#                 kernel_size=self.kernel_sizes[i],
#                 stride=self.strides[i],
#                 padding=self.paddings[i],
#                 bias=False
#             )
#             if i % 2 == 0:
#                 self.conv['act-{0}'.format(i)] = nn_activations[self.act]()

#             if self.apply_maxpools[i] == 1:
#                 self.conv['mp-{0}'.format(i)] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.init_weight()

#     def forward(self, x):
#         rejoin = -1
#         if not x.dtype == torch.long:
#             x = torch.randint(1, 150, size=x.shape)
#         x = self.embedding(x)
#         x = x.permute(0, 2, 1)
#         for layer in self.conv:
#             if isinstance(self.conv[layer], nn.modules.conv.Conv1d):
#                 block = int(layer.split('-')[1])
#                 if block == 0 or self.shortcuts[block - 1] == 1:
#                     rejoin = block + 1
#                     y = x.clone()
#                     if block > 1:
#                         x = F.relu(x)
#             x = self.conv[layer](x)
#             if block == rejoin:
#                 x += y
#                 rejoin = -1
#         x = x.max(2, keepdim=True)[0]
#         x = self.flat(x)
#         x = self.linear_out(x)
#         return x


# for i in range(self.num_layers_conv):
#     if i > 0:
#         self.conv['act-{0}'.format(i)] = nn_activations[self.act]()
#     if i == 0:
#         self.conv['act-{0}'.format(i)] = nn_activations[self.act]()

# for layer in self.conv:
#     if isinstance(self.conv[layer], nn.modules.conv.Conv1d):   # @Sara
#         if block > 0 and self.shortcuts[block - 1] == 1:
#             rejoin = block + 1
#             y = x.clone()
#             count_downsampling = sum(self.apply_maxpools[block:block + 2]) + sum(self.strides[block:block + 2]) - 2
#             for _ in range(count_downsampling):
#                 y = F.avg_pool1d(y, kernel_size=2, ceil_mode=True)
#             y = F.pad(y, (0, 0, self.out_channels[block + 1] - self.out_channels[block - 1], 0))
#     if block == rejoin and 'act' in layer:  # add shortcut to residual just before activation
#         x += y


class Search_Net(Base_Net):
    def __init__(self, input_size=[3, 32, 32], output_size=10, **kw):
        super(Search_Net, self).__init__(input_size, output_size, **kw)
        self.apply_dropouts = kw['apply_dropouts'] if 'apply_dropouts' in kw else (self.num_layers_conv + 1) * net_kws_defaults['apply_dropouts']
        if 'dropout_probs' in kw:
            self.dropout_probs = kw['dropout_probs']
        else:
            self.dropout_probs = np.count_nonzero(self.apply_dropouts) * [net_kws_defaults['dropout_probs'][1]]
            if len(self.apply_dropouts) != 0 and self.apply_dropouts[0] == 1:
                self.dropout_probs[0] = net_kws_defaults['dropout_probs'][0]
        self.apply_dropouts_embedding = self.apply_dropouts[0]
        self.apply_dropouts = self.apply_dropouts[1:]
        if self.apply_dropouts_embedding:
            self.embedding_dropout_prob = self.dropout_probs[0]
            self.embedding_dropout_layer = nn.Dropout(self.embedding_dropout_prob)
            self.dropout_probs = self.dropout_probs[1:]
        self.apply_dropouts_mlp = kw['apply_dropouts_mlp'] if 'apply_dropouts_mlp' in kw else net_kws_defaults['apply_dropouts_mlp']
        self.dropout_probs_mlp = kw['dropout_probs_mlp'] if 'dropout_probs_mlp' in kw else net_kws_defaults['dropout_probs_mlp']
        if self.apply_gap == 1 and self.num_layers_conv > 0:  # GAP is not done when there are no conv layers
            self.conv['gap'] = nn.AdaptiveAvgPool2d(output_size=1)  # this is basically global average pooling, i.e. input of (batch,cin,h,w) is converted to output (batch,cin,1,1)

        dropout_index = 0

        # for i in range(self.num_layers_conv):
        #     self.conv['conv-{0}'.format(i)] = nn.Conv1d(
        #         in_channels=self.embedding_dim if i == 0 else self.out_channels[i - 1],
        #         out_channels=self.out_channels[i],
        #         kernel_size=self.kernel_sizes[i],
        #         stride=self.strides[i],
        #         padding=self.paddings[i],
        #         dilation=self.dilations[i],
        #         groups=self.groups[i],
        #         bias=False
        #     )

        #     if self.apply_maxpools[i] == 1:
        #         self.conv['mp-{0}'.format(i)] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        #     if self.apply_bns[i] == 1:
        #         self.conv['bn-{0}'.format(i)] = nn.BatchNorm2d(self.out_channels[i])

        #     if self.shortcuts[i] == 0:
        #         self.conv['act-{0}'.format(i)] = nn_activations[self.act]()

        #     if self.apply_dropouts[i] == 1:
        #         self.conv['drop-{0}'.format(i)] = nn.Dropout(self.dropout_probs[dropout_index])
        #         dropout_index += 1

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
                self.conv['act-{0}'.format(i)] = nn_activations[self.act]()

            if self.apply_dropouts[i] == 1:
                self.conv['drop-{0}'.format(i)] = nn.Dropout(self.dropout_probs[dropout_index])
                dropout_index += 1

        self.init_weight()

    def forward(self, x):
        rejoin = -1
        if not x.dtype == torch.long:
            x = torch.randint(1, 150, size=x.shape)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        if self.apply_dropouts_embedding:
            x = self.embedding_dropout_layer(x)
        # for layer in self.conv:
        #     if isinstance(self.conv[layer], nn.modules.conv.Conv2d):   # @Sara
        #         block = int(layer.split('-')[1])
        #         if block > 0 and self.shortcuts[block - 1] == 1:
        #             rejoin = block + 1
        #             y = x.clone()
        #             count_downsampling = sum(self.apply_maxpools[block: block + 2]) + sum(self.strides[block: block + 2]) - 2
        #             for _ in range(count_downsampling):
        #                 y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
        #             y = F.pad(y, (0, 0, 0, 0, 0, self.out_channels[block + 1] - self.out_channels[block - 1], 0, 0))
        #     if isinstance(self.conv[layer], nn.modules.conv.Conv1d):   # @Sara
        #         block = int(layer.split('-')[1])
        #         if block == 0 or self.shortcuts[block - 1] == 1:
        #             rejoin = block + 1
        #             y = x.clone()
        #             z = x.clone()
        #             count_downsampling = sum(self.strides[block:block + 2]) - 2
        #             for _ in range(count_downsampling):
        #                 y = F.avg_pool1d(y, kernel_size=2, ceil_mode=True)
        #             y = F.pad(y, (0, 0, self.out_channels[block + 1] - y.shape[1], 0))
        #             if block > 1:
        #                 x = F.relu(x)
        #     x = self.conv[layer](x)
        #     if block == rejoin:  # add shortcut to residual just before activation
        #         while x.shape[2] < y.shape[2]:
        #             y = F.avg_pool1d(y, kernel_size=2, ceil_mode=True)
        #         x += y
        #         rejoin = -1
        for layer in self.conv:
            if isinstance(self.conv[layer], nn.modules.conv.Conv2d):   # @Sara
                block = int(layer.split('-')[1])
                if block > 0 and self.shortcuts[block - 1] == 1:
                    rejoin = block + 1
                    y = x.clone()
                    count_downsampling = sum(self.apply_maxpools[block: block + 2]) + sum(self.strides[block: block + 2]) - 2
                    for _ in range(count_downsampling):
                        y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
                    y = F.pad(y, (0, 0, 0, 0, 0, self.out_channels[block + 1] - self.out_channels[block - 1], 0, 0))
            if isinstance(self.conv[layer], nn.modules.conv.Conv1d):   # @Sara
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


class Net_DPCNN(nn.Module):
    def __init__(self, input_size=[3, 32, 32], output_size=10, **kw):
        super(Net_DPCNN, self).__init__()
        channel_size = 250
        num_blocks = 7
        dropout = [0, 0]
        self.output_size = output_size
        num_classes = output_size
        self.dropout = dropout[0]
        self.num_blocks = num_blocks
        self.top_dropout = dropout[1]
        self.channel_size = channel_size
        self.embeds = nn.Embedding(input_size[0], self.channel_size, padding_idx=0)
        nn.init.normal_(self.embeds.weight, std=0.01)
        nn.init.zeros_(self.embeds.weight[0, :])
        for i in range(self.num_blocks):
            do_skip_1stAct = (i == 0)
            ni = self.channel_size
            no = self.channel_size
            do_downsample = i < self.num_blocks - 1
            exec(f"self.layer{i} = resnet_block({ni}, {no}, {do_skip_1stAct}, do_downsample={do_downsample}, dropout=self.dropout)")
        self.flat = nn.Flatten()
        self.linear_out = nn.Linear(self.channel_size, num_classes)
        nn.init.normal_(self.linear_out.weight, std=0.01)
        nn.init.zeros_(self.linear_out.bias)

    def normal_forward(self, x):
        if not x.dtype == torch.long:
            x = torch.randint(1, 150, size=x.shape)
        x = self.embeds(x)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        x = self.normal_forward(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.max(2, keepdim=True)[0]
        x = self.flat(x)
        if self.top_dropout > 0:
            x = F.dropout(x, self.top_dropout)
        x = self.linear_out(x)
        return x


class resnet_block(nn.Module):
    def __init__(self, ni, no, do_skip_1stAct, pool_padding=1,
                 do_downsample=True, kernel_size=3, dropout=0):
        super(resnet_block, self).__init__()
        self.dropout = dropout
        padding = int((kernel_size - 1) / 2)
        self.pool_padding = pool_padding
        self.do_skip_1stAct = do_skip_1stAct
        self.do_downsample = do_downsample
        self.conv1 = nn.Conv1d(ni, no, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(no, no, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        torch.nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.conv2.weight, mean=0, std=0.01)

    def forward(self, x):
        o = x
        if not self.do_skip_1stAct:
            o = F.relu(o)
        o = F.relu(self.conv1(o))
        if self.dropout > 0:
            o = F.dropout(o, self.dropout)
        o = self.conv2(o)
        o = o + x
        if self.do_downsample:
            if self.pool_padding > 0 or o.size()[-1] > 2:
                o = F._max_pool1d(o, 3, stride=2, padding=self.pool_padding)
        return o


class Search_Net_bigger(Base_Net):
    def __init__(self, input_size=[3, 32, 32], output_size=10, **kw):
        super(Search_Net_bigger, self).__init__(input_size, output_size, **kw)
        self.apply_dropouts = kw['apply_dropouts'] if 'apply_dropouts' in kw else (self.num_layers_conv + 1) * net_kws_defaults['apply_dropouts']
        if 'dropout_probs' in kw:
            self.dropout_probs = kw['dropout_probs']
        else:
            self.dropout_probs = np.count_nonzero(self.apply_dropouts) * [net_kws_defaults['dropout_probs'][1]]
            if len(self.apply_dropouts) != 0 and self.apply_dropouts[0] == 1:
                self.dropout_probs[0] = net_kws_defaults['dropout_probs'][0]
        self.apply_dropouts_embedding = self.apply_dropouts[0]
        self.apply_dropouts = self.apply_dropouts[1:]
        if self.apply_dropouts_embedding:
            self.embedding_dropout_prob = self.dropout_probs[0]
            self.embedding_dropout_layer = nn.Dropout(self.embedding_dropout_prob)
            self.dropout_probs = self.dropout_probs[1:]
        self.apply_dropouts_mlp = kw['apply_dropouts_mlp'] if 'apply_dropouts_mlp' in kw else net_kws_defaults['apply_dropouts_mlp']
        self.dropout_probs_mlp = kw['dropout_probs_mlp'] if 'dropout_probs_mlp' in kw else net_kws_defaults['dropout_probs_mlp']
        dropout_index = 0

        for i in range(self.num_layers_conv):
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

            if self.shortcuts[i] == 1:
                self.conv['act-{0}'.format(i)] = nn_activations[self.act]()

            if self.apply_dropouts[i] == 1:
                self.conv['drop-{0}'.format(i)] = nn.Dropout(self.dropout_probs[dropout_index])
                dropout_index += 1
        self.init_weight()

    def forward(self, x):
        rejoin = -1
        if not x.dtype == torch.long:
            x = torch.randint(1, 150, size=x.shape)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        if self.apply_dropouts_embedding:
            x = self.embedding_dropout_layer(x)
        for layer in self.conv:

            if isinstance(self.conv[layer], nn.modules.conv.Conv2d):
                block = int(layer.split('-')[1])
                if block > 0 and self.shortcuts[block - 1] == 1:
                    rejoin = block + 1
                    y = x.clone()
                    count_downsampling = sum(self.apply_maxpools[block: block + 2]) + sum(self.strides[block: block + 2]) - 2
                    for _ in range(count_downsampling):
                        y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
                    y = F.pad(y, (0, 0, 0, 0, 0, self.out_channels[block + 1] - self.out_channels[block - 1], 0, 0))
            if isinstance(self.conv[layer], nn.modules.conv.Conv1d):
                block = int(layer.split('-')[1])
                if block == 0 or self.shortcuts[block] == 1:
                    rejoin = block + 1
                    y = x.clone()
                    # print("saving y")
                    z = x.clone()
                    count_downsampling = sum(self.strides[block:block + 2]) - 2
                    for _ in range(count_downsampling):
                        y = F.avg_pool1d(y, kernel_size=2, ceil_mode=True)
                    y = F.pad(y, (0, 0, self.out_channels[block + 1] - y.shape[1], 0))
                    if block > 1:
                        # print(f"relu{block}")
                        x = F.relu(x)
            x = self.conv[layer](x)
            # print(layer)
            if block == rejoin:
                while x.shape[2] < y.shape[2]:
                    y = F.avg_pool1d(y, kernel_size=2, ceil_mode=True)
                x += y
                # print("shortcut")
                rejoin = -1
        x = x.max(2, keepdim=True)[0]
        x = self.flat(x)
        if self.apply_dropouts_mlp[0] == 1:
            x = F.dropout(x, p=self.dropout_probs_mlp[0])
        x = self.linear_out(x)
        return x


is_dpcnn = os.environ['is_DPCNN']
Net = Net_DPCNN if is_dpcnn == '1' else Search_Net
