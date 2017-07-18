import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DMax(nn.Module):
    def __init__(self, dimension, windowSize, cuda):
        super(DMax, self).__init__()
        self.dimension = dimension
        self.windowSize = windowSize
        self.cuda = cuda
        self.max_modules = []
        self.gradInput = [torch.FloatTensor()]

    def forward(self, input, sizes):
        # input, sizes = inputs
        while len(self.max_modules) < sizes.size()[0]:
            self.max_modules.append(None)

        output = Variable(torch.FloatTensor(sizes.size()[0], input.size()[1]))
        start_idx = 0
        for i in range(0, sizes.size()[0]):
            max_module = self.max_modules[i]
            if max_module is None:
                if self.cuda:
                    self.max_modules[i] = lambda x: torch.max(x, self.dimension)[0][0].cuda()
                else:  # max will return two vecs, one for value, the other for index, and then (1,N) => N
                    self.max_modules[i] = lambda x: torch.max(x, self.dimension)[0][0]
                max_module = self.max_modules[i]
            output[i] = max_module(input[start_idx: start_idx + sizes[i] - self.windowSize + 1])
            start_idx = start_idx + sizes[i]
        return output


class SoftAttention(nn.Module):
    def __init__(self):
        super(SoftAttention, self).__init__()

    def forward(self, linput, rinput):
        self.lPad = linput.view(-1, linput.size(0), linput.size(1))

        self.lPad = linput  # self.lPad = Padding(0, 0)(linput) TODO: figureout why padding?
        self.M_r = torch.mm(self.lPad, rinput.t())
        self.alpha = F.softmax(self.M_r.transpose(0, 1))
        self.Yl = torch.mm(self.alpha, self.lPad)
        return self.Yl


class LinearLogSoftmax(nn.Module):
    def __init__(self, mem_dim):
        super(LinearLogSoftmax, self).__init__()
        self.layer1 = nn.Linear(mem_dim, 1)

    def forward(self, input):
        var1 = self.layer1(input)
        var1 = var1.view(-1)
        out = F.log_softmax(var1)
        return out


class FullyConnected(nn.Module):
    def __init__(self, emb_dim, mem_dim):
        super(FullyConnected, self).__init__()
        self.emb_dim = emb_dim
        self.mem_dim = mem_dim
        self.linear1 = nn.Linear(self.emb_dim, self.mem_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.mem_dim)
        self.sigmod1 = nn.Sigmoid()
        self.tanh1 = nn.Tanh()

    def forward(self, input):
        i = self.sigmod1(self.linear1(input))
        u = self.tanh1(self.linear2(input))
        out = i.mul(u)  # CMulTable().updateOutput([i, u])
        return out


class SimMul(nn.Module):
    def __init__(self):
        super(SimMul, self).__init__()

    def forward(self, inputa, inputh):  # actually it's a_j vs h_j, element-wise mul
        return inputa.mul(inputh)  # return CMulTable().updateOutput([inputq, inputa])


class TemporalConvoluation(nn.Module):
    def __init__(self, cov_dim, mem_dim, window_size):
        super(TemporalConvoluation, self).__init__()
        self.conv1 = nn.Conv1d(cov_dim, mem_dim, window_size)

    def forward(self, input):
        myinput = input.view(1, input.size()[0], input.size()[1]).transpose(1, 2)  # 1, 150, 56
        output = self.conv1(myinput)[0].transpose(0, 1)  # 56, 150
        return output


class ConvolutionDMax(nn.Module):
    def __init__(self, window_sizes, cov_dim, mem_dim, cuda):
        super(ConvolutionDMax, self).__init__()
        self.window_sizes = window_sizes
        self.cov_dim = cov_dim
        self.mem_dim = mem_dim

        self.tempconvs = nn.ModuleList()
        self.dmaxs = nn.ModuleList()
        for window_size in self.window_sizes:
            self.tempconvs.append(TemporalConvoluation(self.cov_dim, self.mem_dim, window_size))
            self.dmaxs.append(DMax(dimension=0, windowSize=window_size, cuda=cuda))

        self.linear1 = nn.Linear(len(window_sizes) * mem_dim, mem_dim)
        self.relu1 = nn.ReLU()
        self.tanh1 = nn.Tanh()

    def forward(self, input, sizes):
        conv = [None] * len(self.window_sizes)
        pool = [None] * len(self.window_sizes)
        for i, window_size in enumerate(self.window_sizes):
            tempconv = self.tempconvs[i](input)
            conv[i] = self.relu1(tempconv)
            pool[i] = self.dmaxs[i](conv[i], sizes)
        concate = torch.cat(pool, 1)
        linear1 = self.linear1(concate)
        output = self.tanh1(linear1)
        return output
