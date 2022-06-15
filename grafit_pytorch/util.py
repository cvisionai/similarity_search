import torch
from torch.autograd import Function
from torch import nn
import math

class LinearAverageOp(Function):
    @staticmethod
    def forward(ctx, x, y, memory, params):
        T = params[0].item()
        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N
        
        ctx.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(ctx, gradOutput):
        x, memory, y, params = ctx.saved_tensors
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

eps = 1e-8

class NCACrossEntropy(nn.Module): 
    ''' \sum_{j=C} log(p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''
    def __init__(self, labels, margin=0):
        super(NCACrossEntropy, self).__init__()
        self.register_buffer('labels', torch.LongTensor(labels.size(0)))
        self.labels = labels
        self.margin = margin

    def forward(self, x, indexes):
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)
        
        # labels for currect batch
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) 
        same = y.repeat(1, n).eq_(self.labels)

       # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))

        loss = prob_masked.log().sum(0)

        return - loss / batchSize