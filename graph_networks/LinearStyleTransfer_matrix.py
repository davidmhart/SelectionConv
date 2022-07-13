import torch
import torch.nn as nn

from copy import deepcopy

from selectionConv import SelectionConv

class CNN(nn.Module):
    def __init__(self,matrixSize=32):
        super(CNN,self).__init__()

        self.conv1 = SelectionConv(512,256,3,padding_mode="zeros")
        self.conv2 = SelectionConv(256,128,3,padding_mode="zeros")
        self.conv3 = SelectionConv(128,matrixSize,3,padding_mode="zeros")
        self.relu = torch.nn.ReLU()

        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)

    def forward(self,x,edge_index,selections,interp_values=None):
        out = self.relu(self.conv1(x,edge_index,selections,interp_values))
        out = self.relu(self.conv2(out,edge_index,selections,interp_values))
        out = self.conv3(out,edge_index,selections,interp_values)

        n,ch = out.size()
 
        out = torch.mm(out.t(), out).div(n)

        out = out.view(-1)
        return self.fc(out)

class TransformLayer(nn.Module):
    def __init__(self,matrixSize=32):
        super(TransformLayer,self).__init__()
        self.snet = CNN(matrixSize)
        self.cnet = CNN(matrixSize)
        self.matrixSize = matrixSize

        self.compress = SelectionConv(512,matrixSize,1)
        self.unzip = SelectionConv(matrixSize,512,1)

    def forward(self,cF,sF,content_edge_index,content_selections,style_edge_index,style_selections,content_interps=None,style_interps=None,trans=True):
        cMean = torch.mean(cF,dim=0,keepdim=True)
        cF = cF - cMean
        
        sMean = torch.mean(sF,dim=0,keepdim=True)
        sF = sF - sMean

        compress_content = self.compress(cF,content_edge_index,content_selections,content_interps)

        if(trans):
            cMatrix = self.cnet(cF,content_edge_index,content_selections,content_interps)
            sMatrix = self.snet(sF,style_edge_index,style_selections,style_interps)

            sMatrix = sMatrix.view(self.matrixSize,self.matrixSize)
            cMatrix = cMatrix.view(self.matrixSize,self.matrixSize)
            transmatrix = torch.mm(sMatrix,cMatrix)
            transfeature = torch.mm(transmatrix,compress_content.transpose(1,0))
            out = self.unzip(transfeature.transpose(1,0),content_edge_index,content_selections,content_interps)
            out = out + sMean
            return out, transmatrix
        else:
            out = self.unzip(compress_content,content_edge_index,content_selections,content_interps)
            out = out + cMean
            return out
            
    def copy_weights(self, model):
        self.cnet.conv1.copy_weights(model.cnet.convs[0].weight,model.cnet.convs[0].bias)
        self.cnet.conv2.copy_weights(model.cnet.convs[2].weight,model.cnet.convs[2].bias)
        self.cnet.conv3.copy_weights(model.cnet.convs[4].weight,model.cnet.convs[4].bias)
        
        self.snet.conv1.copy_weights(model.snet.convs[0].weight,model.snet.convs[0].bias)
        self.snet.conv2.copy_weights(model.snet.convs[2].weight,model.snet.convs[2].bias)
        self.snet.conv3.copy_weights(model.snet.convs[4].weight,model.snet.convs[4].bias)

        self.cnet.fc = deepcopy(model.cnet.fc)
        self.snet.fc = deepcopy(model.snet.fc)
        
        self.compress.copy_weights(model.compress.weight,model.compress.bias)
        self.unzip.copy_weights(model.unzip.weight,model.unzip.bias)
