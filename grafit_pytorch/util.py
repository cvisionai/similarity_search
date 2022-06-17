import torch
from torch.autograd import Function
from torch import nn
import math
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torch.nn.functional as F

class LinearAverageOp(Function):
    @staticmethod
    def forward(ctx, x, y, memory, params):
        T = params[0]
        # inner product
        #print(x)
        out = torch.mm(F.normalize(x,dim=1,p=2), memory.t())
        out.div_(T) # batchSize * N
        
        ctx.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(ctx, gradOutput):
        x, memory, y, params = ctx.saved_tensors
        T = params[0]
        momentum = params[1]
        # add temperature
        gradOutput.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput, memory)
        gradInput.resize_as_(x)

        #print(gradInput)
        #print(f"GradInput shape: {gradInput.shape}")
        # update the non-parametric data
        weight_pos = memory.index_select(0, y.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        #print(gradInput)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize
        self.register_buffer('params',torch.tensor([T, momentum],requires_grad=False))
        stdv = 1. / math.sqrt(inputSize/3)
        #self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv).to("cuda"))
        init = torch.normal(0., 0.01, size=(outputSize, inputSize),requires_grad=False)
        init_norm = init.pow(2).sum(1,keepdim=True).pow(0.5)
        init = init.div(init_norm)
        self.register_buffer('memory', init)

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

eps = 0.01

class NCACrossEntropy(nn.Module): 
    ''' \sum_{j=C} log(p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''
    def __init__(self, labels, margin=0):
        super(NCACrossEntropy, self).__init__()
        self.register_buffer('labels', torch.tensor(labels.size(0),requires_grad=False, device=torch.device('cuda:0'),dtype=torch.long))
        self.labels = labels
        #print(self.labels)
        self.margin = margin

    def forward(self, x, indexes):
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)
        # labels for currect batch
        #print(f"Printing current batch indices: {indexes}")
        y = torch.index_select(self.labels, 0, indexes).view(batchSize, 1) 
        same = y.repeat(1, n).eq_(self.labels)

       # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        prob = torch.div(p, Z)
        #min_prob = torch.tensor([eps for i in range(len(prob))],dtype=torch.float,requires_grad=False,device=torch.devic("cuda"))
        #prob = torch.max(prob,min_prob)
        #print(prob)
        prob_masked = torch.masked_select(prob, prob.ne(0))
        #print(prob_masked)
        loss = prob_masked.log().sum(0)
        return - loss / batchSize

class FathomnetDataset(Dataset):
    """Data Set for loading Fathomnet ROIs"""
    def __init__(self, csv_file, root_dir, transform=None, start_idx=None):
        self.root_dir = root_dir
        self.rois = pd.read_csv(csv_file)
        if start_idx is not None:
            self.rois = self.rois.loc[start_idx:]
        self.transform = transform

    def __len__(self):
        return (len(self.rois))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.rois.iloc[idx,0])
        #print(img_name)
        x1,y1,x2,y2 = self.rois.iloc[idx,1:5]
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Bounds checking
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > image.width:
            x2 = image.width
        if y2 > image.height:
            y2 = image.height

        assert(x1 < x2 and y1 < y2), f"Bad ROI dimensions: {x1,y1,x2,y2} in {img_name}"

        image = image.crop((x1,y1,x2,y2))
        image = self.transform(image)
        sample = {"image" : image, "name" : self.rois.iloc[idx,0], "label" : self.rois.iloc[idx,5], "roi" : {"x1" : x1,"y1" : y1, "x2" : x2, "y2" : y2}, "index" : idx}

        return sample
    
    def get_labels(self):
        labels = self.rois.iloc[:,5].to_list()
        return labels