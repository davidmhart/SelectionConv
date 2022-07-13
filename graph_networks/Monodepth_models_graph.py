from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from selectionConv import SelectionConv
from pooling import unpoolBilinear, stridePoolCluster, maxPoolKernel
from math import sqrt

def copyBatchNorm(target,source):
    target.weight = source.weight
    target.bias = source.bias
    target.running_mean = source.running_mean
    target.running_var = source.running_var
    target.eps = source.eps


class SimpleBatchNorm():
    def __init__(self):
        self.gamma = 1
        self.beta = 0
        self.running_mean = 0
        self.running_var = 1
        self.epsilon = 1e-4

    def __call__(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon) * self.gamma + self.beta

    def set_values(self, layer):
        self.gamma = layer.weight
        self.beta = layer.bias
        self.running_mean = layer.running_mean
        self.running_var = layer.running_var
        self.epsilon = layer.eps


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(conv, self).__init__()
        self.conv_base = SelectionConv(num_in_layers, num_out_layers, kernel_size=kernel_size)
        self.normalize = nn.BatchNorm1d(num_out_layers)

    def forward(self, x, edge_index, selections, cluster = None):
        x = self.conv_base(x, edge_index, selections)
        if cluster is not None:
            x = stridePoolCluster(x,cluster)
        x = self.normalize(x)
        return F.elu(x, inplace=True)
        
    def copy_weights(self,layer):
        self.conv_base.copy_weights(layer.conv_base.weight,layer.conv_base.bias)
        #self.normalize.set_values(layer.normalize)
        copyBatchNorm(self.normalize,layer.normalize)

class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, strided = False):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.conv1 = conv(num_in_layers, num_out_layers, 3)
        self.conv2 = conv(num_out_layers, num_out_layers, 3)
        self.conv3 = SelectionConv(num_in_layers, num_out_layers, kernel_size=1)
        self.normalize = nn.BatchNorm1d(num_out_layers)
        self.strided = strided

    def forward(self, x, edge_index, selections, cluster = None, down_edge_index = None, down_selections = None):
        shortcut = []
        x_out = self.conv1(x, edge_index, selections)
        if self.strided:
            x_out = stridePoolCluster(x_out,cluster)
            x_out = self.conv2(x_out, down_edge_index, down_selections)
            shortcut = self.conv3(x, edge_index, selections)
            shortcut = stridePoolCluster(shortcut,cluster)
        else:
            x_out = self.conv2(x_out, edge_index, selections)
            shortcut = self.conv3(x, edge_index, selections)

        return F.elu(self.normalize(x_out + shortcut), inplace=True)
        
    def copy_weights(self,layer):
        self.conv1.copy_weights(layer.conv1)
        self.conv2.copy_weights(layer.conv2)
        self.conv3.copy_weights(layer.conv3.weight,layer.conv3.bias)
        #self.normalize.set_values(layer.normalize)
        copyBatchNorm(self.normalize,layer.normalize)

def resblock_basic(num_in_layers, num_out_layers):
    layers = []
    firstconv = resconv_basic(num_in_layers, num_out_layers, True)
    secondconv = resconv_basic(num_out_layers, num_out_layers)
    return firstconv, secondconv


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(upconv, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size)

    def forward(self, x, cluster, up_edge_index, up_selections):
        x = unpoolBilinear(x,cluster,up_edge_index,up_selections)
        return self.conv1(x, up_edge_index, up_selections)
        
    def copy_weights(self,layer):
        self.conv1.copy_weights(layer.conv1)


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = SelectionConv(num_in_layers, 2, kernel_size=3)
        self.normalize = nn.BatchNorm1d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, selections):
        x = self.conv1(x, edge_index, selections)
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)
        
    def copy_weights(self,layer):
        self.conv1.copy_weights(layer.conv1.weight,layer.conv1.bias)
        #self.normalize.set_values(layer.normalize)
        copyBatchNorm(self.normalize,layer.normalize)

class Resnet18_graph(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet18_graph, self).__init__()
        # encoder
        self.conv1 = conv(num_in_layers, 64, 7)  # H/2  -   64D
        self.downconv2, self.conv2 = resblock_basic(64, 64)  # H/8  -  64D
        self.downconv3, self.conv3 = resblock_basic(64, 128)  # H/16 -  128D
        self.downconv4, self.conv4 = resblock_basic(128, 256)  # H/32 - 256D
        self.downconv5, self.conv5 = resblock_basic(256, 512)  # H/64 - 512D

        # decoder
        self.upconv6 = upconv(512, 512, 3)
        self.iconv6 = conv(256+512, 512, 3)

        self.upconv5 = upconv(512, 256, 3)
        self.iconv5 = conv(128+256, 256, 3)

        self.upconv4 = upconv(256, 128, 3)
        self.iconv4 = conv(64+128, 128, 3)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3)
        self.iconv3 = conv(64+64 + 2, 64, 3)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3)
        self.iconv2 = conv(64+32 + 2, 32, 3)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3)
        self.iconv1 = conv(16+2, 16, 3)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, clusters, edge_indexes, selections_list):
        
        # Strided Convolution
        x1 = self.conv1(x,edge_indexes[0],selections_list[0],clusters[0])
        
        x_pool1 = maxPoolKernel(x1,edge_indexes[1],selections_list[1],clusters[1],kernel_size=3)
        x2 = self.downconv2(x_pool1,edge_indexes[2],selections_list[2],clusters[2],edge_indexes[3],selections_list[3])
        x2 = self.conv2(x2,edge_indexes[3],selections_list[3])
        x3 = self.downconv3(x2,edge_indexes[3],selections_list[3],clusters[3],edge_indexes[4],selections_list[4])
        x3 = self.conv3(x3,edge_indexes[4],selections_list[4])
        x4 = self.downconv4(x3,edge_indexes[4],selections_list[4],clusters[4],edge_indexes[5],selections_list[5])
        x4 = self.conv4(x4,edge_indexes[5],selections_list[5])
        x5 = self.downconv5(x4,edge_indexes[5],selections_list[5],clusters[5],edge_indexes[6],selections_list[6])
        x5 = self.conv5(x5,edge_indexes[6],selections_list[6])

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5, clusters[5], edge_indexes[5], selections_list[5])
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6, edge_indexes[5], selections_list[5])

        upconv5 = self.upconv5(iconv6,clusters[4], edge_indexes[4], selections_list[4])
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5, edge_indexes[4], selections_list[4])

        upconv4 = self.upconv4(iconv5, clusters[3], edge_indexes[3], selections_list[3])
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4, edge_indexes[3], selections_list[3])
        self.disp4 = self.disp4_layer(iconv4, edge_indexes[3], selections_list[3])
        self.udisp4 = unpoolBilinear(self.disp4, clusters[2], edge_indexes[2], selections_list[2])

        upconv3 = self.upconv3(iconv4, clusters[2], edge_indexes[2], selections_list[2])
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3, edge_indexes[2], selections_list[2])
        self.disp3 = self.disp3_layer(iconv3, edge_indexes[2], selections_list[2])
        self.udisp3 = unpoolBilinear(self.disp3, clusters[1], edge_indexes[1], selections_list[1])

        upconv2 = self.upconv2(iconv3, clusters[1], edge_indexes[1], selections_list[1])
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2, edge_indexes[1], selections_list[1])
        self.disp2 = self.disp2_layer(iconv2, edge_indexes[1], selections_list[1])
        self.udisp2 = unpoolBilinear(self.disp2, clusters[0], edge_indexes[0], selections_list[0])

        upconv1 = self.upconv1(iconv2, clusters[0], edge_indexes[0], selections_list[0])
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1, edge_indexes[0], selections_list[0])
        self.disp1 = self.disp1_layer(iconv1, edge_indexes[0], selections_list[0])
        return self.disp1, self.disp2, self.disp3, self.disp4

    def copy_weights(self, model):
        self.conv1.copy_weights(model.conv1)
        self.downconv2.copy_weights(model.conv2[0])
        self.conv2.copy_weights(model.conv2[1])
        self.downconv3.copy_weights(model.conv3[0])
        self.conv3.copy_weights(model.conv3[1])
        self.downconv4.copy_weights(model.conv4[0])
        self.conv4.copy_weights(model.conv4[1])
        self.downconv5.copy_weights(model.conv5[0])
        self.conv5.copy_weights(model.conv5[1])
        
        self.upconv6.copy_weights(model.upconv6)
        self.iconv6.copy_weights(model.iconv6)
        self.upconv5.copy_weights(model.upconv5)
        self.iconv5.copy_weights(model.iconv5)
        self.upconv4.copy_weights(model.upconv4)
        self.iconv4.copy_weights(model.iconv4)
        self.upconv3.copy_weights(model.upconv3)
        self.iconv3.copy_weights(model.iconv3)
        self.upconv2.copy_weights(model.upconv2)
        self.iconv2.copy_weights(model.iconv2)
        self.upconv1.copy_weights(model.upconv1)
        self.iconv1.copy_weights(model.iconv1)
        
        self.disp4_layer.copy_weights(model.disp4_layer)
        self.disp3_layer.copy_weights(model.disp3_layer)
        self.disp2_layer.copy_weights(model.disp2_layer)
        self.disp1_layer.copy_weights(model.disp1_layer)


