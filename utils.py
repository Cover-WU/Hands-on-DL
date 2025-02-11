import time
from torch import nn
import d2l.torch as d2l
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        ''' add the args into the data '''
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        ''' Reset the data to zero '''
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def try_gpu(i=0): 
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # compare the predicted class with the true label, convert the dtype before comparison
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear', 
                 fmts=('-', 'm--', 'g-', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts
    
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y) # number of indicators
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes() # set the aesthetic parameters of the plot
        display.display(self.fig) # display the plot
        display.clear_output(wait=True) # clear the output for the next plot

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

def grad_clipping(net, theta):
    '''
    clip the gradient to stablize the training
    '''
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        # 如果是嵌套列表，展开成单个列表
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # 统计词频
        counter = {}
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1
        # 保留的特殊符号
        self.reserved_tokens = reserved_tokens if reserved_tokens else ['<unk>']
        # 排序词表
        self.token_freqs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        # 构建词表
        self.idx_to_token = self.reserved_tokens + [token for token, freq in self.token_freqs if freq >= min_freq]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # 返回 '<unk>' 索引，如果找不到该 token
            return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


if __name__ == '__main__':
    pass