import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
from torch import nn
from selectionConv import SelectionConv
from pooling import maxPoolCluster, unpoolCluster

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def copyBatchNorm(target,source):
    target.weight = source.weight
    target.bias = source.bias
    target.running_mean = source.running_mean
    target.running_var = source.running_var
    target.eps = source.eps

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()

        self.encode = _ResBlock(in_channels,in_channels,out_channels)

    def forward(self, x, cluster, down_edge_index, down_selections, down_interps):
        x = maxPoolCluster(x,cluster)
        return self.encode(x,down_edge_index,down_selections,down_interps)

    def copy_weights(self,layer):
        self.encode.copy_weights(layer.encode[1])

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = _ResBlock(2*in_channels, out_channels, out_channels)

    def forward(self, x, skip_connection,up_cluster,up_edge_index,up_selections,up_interps):
        x = unpoolCluster(x,up_cluster)
        x = torch.cat((x,skip_connection),dim=1)
        return self.decode(x,up_edge_index,up_selections,up_interps)

    def copy_weights(self,layer):
        self.decode.copy_weights(layer.decode)
    
class _ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_ResBlock, self).__init__()
        self.conv1 = SelectionConv(in_channels, middle_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.conv2 = SelectionConv(middle_channels, middle_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(middle_channels)
        self.conv3 = SelectionConv(middle_channels, out_channels, kernel_size=1)

        self.res_conv = SelectionConv(in_channels, out_channels, kernel_size=1)
        self.res_bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, edge_index, selections, interps):
        out = self.conv1(x,edge_index,selections,interps)
        out = self.relu(self.bn1(out))
        out = self.conv2(out,edge_index,selections,interps)
        out = self.relu(self.bn2(out))
        out = self.conv3(out,edge_index,selections,interps)
        
        return self.relu(out + self.res_bn(self.res_conv(x,edge_index,selections,interps)))
                         
    def copy_weights(self,layer):
        self.conv1.copy_weights(layer.net[0].weight, layer.net[0].bias)
        self.conv2.copy_weights(layer.net[3].weight, layer.net[3].bias)
        self.conv3.copy_weights(layer.net[6].weight, layer.net[6].bias)
        copyBatchNorm(self.bn1,layer.net[1])
        copyBatchNorm(self.bn2,layer.net[4])
        
        self.res_conv.copy_weights(layer.res[0].weight, layer.res[0].bias)
        copyBatchNorm(self.res_bn,layer.res[1])

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, feat=32):
        super(UNet, self).__init__()
        self.start = SelectionConv(in_channels, feat, kernel_size=3)
        self.enc1 = _EncoderBlock(feat, feat*2)
        self.enc2 = _EncoderBlock(feat*2, feat*4)
        self.enc3 = _EncoderBlock(feat*4, feat*8)
        self.enc4 = _EncoderBlock(feat*8, feat*16)
        self.enc5 = _EncoderBlock(feat*16, feat*16)
        self.dec5 = _DecoderBlock(feat*16, feat*8)
        self.dec4 = _DecoderBlock(feat*8, feat*4)
        self.dec3 = _DecoderBlock(feat*4, feat*2)
        self.dec2 = _DecoderBlock(feat*2, feat)
        self.dec1 = _DecoderBlock(feat,feat)
        self.final = SelectionConv(feat, num_classes, kernel_size=3)
        initialize_weights(self)

    def forward(self, graph):
        enc1 = self.start(graph.x,graph.edge_indexes[0],graph.selections_list[0],graph.interps_list[0] if hasattr(graph,"interps_list") else None)
        enc2 = self.enc1(enc1,graph.clusters[0],graph.edge_indexes[1],graph.selections_list[1],graph.interps_list[1] if hasattr(graph,"interps_list") else None)
        enc3 = self.enc2(enc2,graph.clusters[1],graph.edge_indexes[2],graph.selections_list[2],graph.interps_list[2] if hasattr(graph,"interps_list") else None)
        enc4 = self.enc3(enc3,graph.clusters[2],graph.edge_indexes[3],graph.selections_list[3],graph.interps_list[3] if hasattr(graph,"interps_list") else None)
        enc5 = self.enc4(enc4,graph.clusters[3],graph.edge_indexes[4],graph.selections_list[4],graph.interps_list[4] if hasattr(graph,"interps_list") else None)
        center = self.enc5(enc5,graph.clusters[4],graph.edge_indexes[5],graph.selections_list[5],graph.interps_list[5] if hasattr(graph,"interps_list") else None)
        dec5 = self.dec5(center,enc5,graph.clusters[4],graph.edge_indexes[4],graph.selections_list[4],graph.interps_list[4] if hasattr(graph,"interps_list") else None)
        dec4 = self.dec4(dec5,enc4,graph.clusters[3],graph.edge_indexes[3],graph.selections_list[3],graph.interps_list[3] if hasattr(graph,"interps_list") else None)
        dec3 = self.dec3(dec4,enc3,graph.clusters[2],graph.edge_indexes[2],graph.selections_list[2],graph.interps_list[2] if hasattr(graph,"interps_list") else None)
        dec2 = self.dec2(dec3,enc2,graph.clusters[1],graph.edge_indexes[1],graph.selections_list[1],graph.interps_list[1] if hasattr(graph,"interps_list") else None)
        dec1 = self.dec1(dec2,enc1,graph.clusters[0],graph.edge_indexes[0],graph.selections_list[0],graph.interps_list[0] if hasattr(graph,"interps_list") else None)
        final = self.final(dec1,graph.edge_indexes[0],graph.selections_list[0],graph.interps_list[0] if hasattr(graph,"interps_list") else None)
        return final #, [enc1,enc2,enc3,enc4,enc5,center,dec5,dec4,dec3,dec2,dec1]
    
    def copy_weights(self,net):
        self.start.copy_weights(net.start.weight,net.start.bias)
        self.enc1.copy_weights(net.enc1)
        self.enc2.copy_weights(net.enc2)
        self.enc3.copy_weights(net.enc3)
        self.enc4.copy_weights(net.enc4)
        self.enc5.copy_weights(net.enc5)
        self.dec1.copy_weights(net.dec1)
        self.dec2.copy_weights(net.dec2)
        self.dec3.copy_weights(net.dec3)
        self.dec4.copy_weights(net.dec4)
        self.dec5.copy_weights(net.dec5)
        self.final.copy_weights(net.final.weight,net.final.bias)
