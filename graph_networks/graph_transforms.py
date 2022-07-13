""" graph_transforms.py

This is the implementation of transforming a traditional CNN
to a SelectionConv-based graph CNN
So far this is just used for segmentation
"""
from copy import deepcopy
from typing import Dict, Iterable, OrderedDict, Tuple, Union #,Literal     Only supported in Python 3.8+
from typing_extensions import Literal
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchvision.models.segmentation.fcn import FCN

from selectionConv import SelectionConv
import pooling as P

def transform_network(network: nn.Module):
    """ Transforms a neural network from a tensor based network to a graph based network

    Parameters
    ----------
    - network: the network to transform

    Returns
    -------
    - the transformed network
    """
    network = deepcopy(network)
    if type(network) in __MAPPING:
        with torch.no_grad():
            return __MAPPING[type(network)].from_torch(network)
    if not isinstance(network, nn.Module):
        raise ValueError(f"Must be of type Module but got: {type(network)}")
    if not list(network.children()):
        raise NotImplementedError(f"{type(network)} is not implemented yet")
    for name, child in network.named_children():
        transformed_child = transform_network(child)
        setattr(network, name, transformed_child)
    return network


class GraphTracker:
    """ a wrapper around the graph data for easily overwriting the forward function of existing modules.

    Parameters
    ----------
    - graph: the graph data
    - level: the current depth the graph is being operated on
    - x: the node data at the current level
    """
    def __init__(self, graph, x=None, level=0):
        self.graph = graph
        self.x = graph.x if x is None else x
        self.level = level
        
    def from_x(self, x):
        """ create the same graph with different node values"""
        return GraphTracker(self.graph,x,level=self.level)
    
    def edge_index(self):
        return self.graph.edge_indexes[self.level]
    
    def selections(self):
        return self.graph.selections_list[self.level]
    
    def interps(self):
        if hasattr(self.graph,"interps_list"):
            return self.graph.interps_list[self.level]
        else:
            return None
    
    def cluster(self):
        return self.graph.clusters[self.level]
    
    def __iadd__(self, other):
        self.x = self.x + other.x
        return self

    def __repr__(self):
        return f"GraphTracker(x={tuple(self.x.shape)},level={self.level})"

    

def _single(pair, name):
    """ converts a tuple into a single number

    Parameters
    ----------
    - pair: the potential pair of values
    - name: the name of the values for more readable errors

    Returns
    -------
    - the single value
    """
    if isinstance(pair, int):
        return pair
    if not isinstance(pair, tuple):
        raise ValueError(f"{name} must either be int or tuple but got: {type(pair)}")
    if len(pair) != 2:
        raise ValueError(f"{name} must be a 2-tuple but got: {pair}")
    if pair[0] != pair[1]:
        raise ValueError(f"{name} must be a square tuple")
    return pair[0]


class SelModule(nn.Module):
    """ A super class for all graph based modules to inherit from
    """
    @classmethod
    def from_torch(cls, network):
        """ creates a new graph based module from an existing 2d based module and copies weights accordingly.  Each child class should implement this method

        Parameters
        ----------
        - network: the existing 2d based module

        Returns
        -------
        - the new graph based module
        """
        raise NotImplementedError


class SelConv(SelModule, nn.modules.conv._ConvNd):
    """ A wrapper class around the SelectionConv class that allows for easy
    use in a transformed network

    Parameters
    ----------
    - in_channels: the number of incoming channels
    - out_channels: the number of outgoing channels
    - kernel_size: the size of the convolution kernel
    - stride: the stride at which to perform convolution
    - padding: the amount of padding to be used
    - dilation: the dilation of the kernel
    - groups: the groups of filters for the convolution
    - bias: whether or not to include a bias
    - padding_mode: the type of padding to be used
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]]=1,
            padding: Union[int, Tuple[int, int]]=0,
            dilation: Union[int, Tuple[int, int]]=1,
            groups: int = 1,
            bias: bool=True,
            padding_mode: str='zeros',
            device=None,
            dtype=None,
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, (0,), groups, bias, padding_mode, **factory_kwargs)
        self.single_stride = _single(stride, "stride")
        if self.single_stride not in (1, 2):
            raise NotImplementedError(f"Only strides of 1 and 2 are supported but got {stride}")
        self.conv_operation = SelectionConv(
            in_channels,
            out_channels,
            _single(kernel_size, "kernel_size"),
            _single(dilation, "dilation"),
            padding_mode,
        )

    def forward(self, inputs: GraphTracker):
        x = self.conv_operation(inputs.x, inputs.edge_index(), inputs.selections(), inputs.interps())
        ret = inputs.from_x(x)
        if self.single_stride == 2:
            x = P.stridePoolCluster(x, ret.cluster())
            ret.x = x
            ret.level += 1
        return ret

    @classmethod
    def from_torch(cls, network):
        ret = SelConv(
            network.in_channels,
            network.out_channels,
            network.kernel_size,
            network.stride,
            network.padding,
            network.dilation,
            network.groups,
            network.bias is not None,
            network.padding_mode,
        )
        ret.conv_operation.copy_weights(network.weight, network.bias)
        return ret

class SelMaxPool(SelModule):
    """ A graph based max pool module

    Parameters
    ----------
    - kernel_size: the size of the maxpool kernel
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, inputs):
        x = P.maxPoolKernel(inputs.x, inputs.edge_index(), inputs.selections(), inputs.cluster(), self.kernel_size)
        ret = inputs.from_x(x)
        ret.level += 1
        return ret

    @classmethod
    def from_torch(cls, network):
        ret = SelMaxPool(network.kernel_size)
        return ret

class SelBatchNorm(SelModule):
    """ A graph based BatchNorm module
    """
    def __init__(self,num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        #self.bn = SimpleBatchNorm()

    def forward(self, inputs):
        x = self.bn(inputs.x)
        ret = inputs.from_x(x)
        return ret
    
    def copyBatchNorm(self,source):
        self.bn.weight = source.weight
        self.bn.bias = source.bias
        self.bn.running_mean = source.running_mean
        self.bn.running_var = source.running_var
        self.bn.eps = source.eps
    
    @classmethod
    def from_torch(cls, network):
        ret = SelBatchNorm(network.num_features)
        ret.copyBatchNorm(network)
        #ret.bn.set_values(network)
        return ret

class SelReLU(SelModule):
    """ A graph based ReLU module

    Parameters
    ----------
    - inplace: whether or not to perform relu in place
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs):
        if self.inplace:
            inputs.x = F.relu(inputs.x, self.inplace)
            return inputs
        else:
            x = F.relu(inputs.x, self.inplace)
            ret = inputs.from_x(x)
            return ret

    @classmethod
    def from_torch(cls, network):
        return SelReLU(network.inplace)


class SelSequential(SelModule, nn.Sequential):
    """ A graph based Sequential module
    """
    @classmethod
    def from_torch(cls, network: nn.Sequential):
        return SelSequential(*map(transform_network, network))


class SelDropout(SelModule):
    """ A graph based dropout module
    """
    def forward(self, inputs):
        return inputs

    @classmethod
    def from_torch(cls, network):
        return SelDropout()

def sel_binlinear_interp(
        inputs: GraphTracker,
        up_or_down: Literal["up", "down"]="up",
    ) -> GraphTracker:
    """ Performs bilinear interpolation as a single cluster step

    Parameters
    ----------
    - inputs: the input graph
    - up_or_down: either "up" or "down" indicating if it is upsampling or downsampling

    Returns
    -------
    - the interpolated graph
    """
    supported_up_or_downs = ("up", "down")
    if up_or_down not in supported_up_or_downs:
        raise ValueError(f"up_or_down must either be 'up' or 'down' not: {up_or_down}")
    ret = inputs.from_x(inputs.x)
    dx = -1 if up_or_down == "up" else 1
    ret.level += dx
    cluster = ret.cluster()
    up_edge_index = ret.edge_index()
    
    #up_selections = ret.selections()
    #ret.x = P.unpoolBilinear(ret.x, cluster, up_edge_index, up_selections)
    
    up_interps = ret.interps()
    ret.x = P.unpoolInterpolated(ret.x,cluster,up_edge_index,up_interps)
    
    #ret.x = P.unpoolCluster(inputs.x, inputs.clusters[inputs.cluster_id])
    return ret


def sel_interpolate(
        inputs: GraphTracker,
        target_level: int,
    ) -> GraphTracker:
    """ interpolates a graph to a given cluster_id

    Parameters
    ----------
    - inputs: the input graph data
    - target_cluster_id: the target cluster

    Returns
    -------
    - the interpolated graph
    """
    up_or_down = "up" if target_level < inputs.level else "down"
    while inputs.level != target_level:
        inputs = sel_binlinear_interp(inputs, up_or_down)
    return inputs


class SelSimpleSegmentationModel(SelModule):
    """ A graph version of the simple segmentation model defined in torchvision's segmentation model.  This is needed since the interpolate function we use needs different parameters than what is used in torch.
    """
    __constants__ = ["aux_classifier"]
    def __init__(self, backbone, classifier, aux_classifier = None):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: GraphTracker) -> Dict[str, GraphTracker]:
        starting_level = x.level
        features = self.backbone(x)
        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = sel_interpolate(x, starting_level)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = sel_interpolate(x, starting_level)
            result["aux"] = x
        return result

    @classmethod
    def from_torch(cls, network):
        ret = SelSimpleSegmentationModel(
            backbone = transform_network(network.backbone),
            classifier = transform_network(network.classifier),
            aux_classifier=transform_network(network.aux_classifier) if network.aux_classifier is not None else None,
        )
        return ret


__MAPPING = {
    nn.Conv2d: SelConv,
    nn.BatchNorm2d: SelBatchNorm,
    nn.ReLU: SelReLU,
    nn.Sequential: SelSequential,
    nn.Dropout: SelDropout,
    nn.MaxPool2d: SelMaxPool,
    FCN: SelSimpleSegmentationModel,
}
