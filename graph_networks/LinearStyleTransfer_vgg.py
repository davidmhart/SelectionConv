import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from selectionConv import SelectionConv

from pooling import unpoolCluster, maxPoolCluster

class encoder(torch.nn.Module):

    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = SelectionConv(3,3,1,padding_mode="reflect")
        self.conv2 = SelectionConv(3,64,3,padding_mode="reflect")
        self.conv3 = SelectionConv(64,64,3,padding_mode="reflect")
        
        self.conv4 = SelectionConv(64,128,3,padding_mode="reflect")
        self.conv5 = SelectionConv(128,128,3,padding_mode="reflect")
        
        self.conv6 = SelectionConv(128,256,3,padding_mode="reflect")
        self.conv7 = SelectionConv(256,256,3,padding_mode="reflect")
        self.conv8 = SelectionConv(256,256,3,padding_mode="reflect")
        self.conv9 = SelectionConv(256,256,3,padding_mode="reflect")
        
        self.conv10 = SelectionConv(256,512,3,padding_mode="reflect")
        
        self.relu = torch.nn.ReLU()

    def copy_weights(self, model):
        self.conv1.copy_weights(model.conv1.weight,model.conv1.bias)
        self.conv2.copy_weights(model.conv2.weight,model.conv2.bias)
        self.conv3.copy_weights(model.conv3.weight,model.conv3.bias)
        self.conv4.copy_weights(model.conv4.weight,model.conv4.bias)
        self.conv5.copy_weights(model.conv5.weight,model.conv5.bias)
        self.conv6.copy_weights(model.conv6.weight,model.conv6.bias)
        self.conv7.copy_weights(model.conv7.weight,model.conv7.bias)
        self.conv8.copy_weights(model.conv8.weight,model.conv8.bias)
        self.conv9.copy_weights(model.conv9.weight,model.conv9.bias)
        self.conv10.copy_weights(model.conv10.weight,model.conv10.bias)


    def forward(self,graph):
        
        edge_index = graph.edge_indexes[0]
        selections = graph.selections_list[0]
        interps = graph.interps_list[0] if hasattr(graph,'interps_list') else None
        
        output = {}
        out = self.conv1(graph.x,edge_index,selections,interps)
        out = self.conv2(out,edge_index,selections,interps)
        output['r11'] = self.relu(out)

        out = self.conv3(output['r11'],edge_index,selections,interps)
        output['r12'] = self.relu(out)

        output['p1'] = maxPoolCluster(output['r12'],graph.clusters[0])
        edge_index = graph.edge_indexes[1]
        selections = graph.selections_list[1]
        interps = graph.interps_list[1] if hasattr(graph,'interps_list') else None
        
        out = self.conv4(output['p1'],edge_index,selections,interps)
        output['r21'] = self.relu(out)

        out = self.conv5(output['r21'],edge_index,selections,interps)
        output['r22'] = self.relu(out)

        output['p2'] = maxPoolCluster(output['r22'],graph.clusters[1])
        edge_index = graph.edge_indexes[2]
        selections = graph.selections_list[2]
        interps = graph.interps_list[2] if hasattr(graph,'interps_list') else None
        
        out = self.conv6(output['p2'],edge_index,selections,interps)
        output['r31'] = self.relu(out)
        #if(matrix31 is not None):
        #    feature3,transmatrix3 = matrix31(output['r31'],sF['r31'])
        #    out = self.reflecPad7(feature3)
        #else:
        #    out = self.reflecPad7()
        out = self.conv7(output['r31'],edge_index,selections,interps)
        output['r32'] = self.relu(out)

        out = self.conv8(output['r32'],edge_index,selections,interps)
        output['r33'] = self.relu(out)

        out = self.conv9(output['r33'],edge_index,selections,interps)
        output['r34'] = self.relu(out)

        output['p3'] = maxPoolCluster(output['r34'],graph.clusters[2])
        edge_index = graph.edge_indexes[3]
        selections = graph.selections_list[3]
        interps = graph.interps_list[3] if hasattr(graph,'interps_list') else None

        out = self.conv10(output['p3'],edge_index,selections,interps)
        output['r41'] = self.relu(out)

        return output

class decoder(torch.nn.Module):

    def __init__(self):
        super(decoder, self).__init__()
        self.conv11 = SelectionConv(512,256,3,padding_mode="reflect")
        
        self.conv12 = SelectionConv(256,256,3,padding_mode="reflect")
        self.conv13 = SelectionConv(256,256,3,padding_mode="reflect")
        self.conv14 = SelectionConv(256,256,3,padding_mode="reflect")
        self.conv15 = SelectionConv(256,128,3,padding_mode="reflect")
        
        self.conv16 = SelectionConv(128,128,3,padding_mode="reflect")
        self.conv17 = SelectionConv(128,64,3,padding_mode="reflect")
        
        self.conv18 = SelectionConv(64,64,3,padding_mode="reflect")
        self.conv19 = SelectionConv(64,3,3,padding_mode="reflect")
        
        self.relu = torch.nn.ReLU()

    def copy_weights(self, model):
        self.conv11.copy_weights(model.conv11.weight,model.conv11.bias)
        self.conv12.copy_weights(model.conv12.weight,model.conv12.bias)
        self.conv13.copy_weights(model.conv13.weight,model.conv13.bias)
        self.conv14.copy_weights(model.conv14.weight,model.conv14.bias)
        self.conv15.copy_weights(model.conv15.weight,model.conv15.bias)
        self.conv16.copy_weights(model.conv16.weight,model.conv16.bias)
        self.conv17.copy_weights(model.conv17.weight,model.conv17.bias)
        self.conv18.copy_weights(model.conv18.weight,model.conv18.bias)
        self.conv19.copy_weights(model.conv19.weight,model.conv19.bias)


    def forward(self,x,graph):

        edge_index = graph.edge_indexes[3]
        selections = graph.selections_list[3]
        interps = graph.interps_list[3] if hasattr(graph,'interps_list') else None
        
        out = self.conv11(x,edge_index,selections,interps)
        out = self.relu(out)
              
        out = unpoolCluster(out,graph.clusters[2])
        edge_index = graph.edge_indexes[2]
        selections = graph.selections_list[2]
        interps = graph.interps_list[2] if hasattr(graph,'interps_list') else None
        
        out = self.conv12(out,edge_index,selections,interps)
        out = self.relu(out)
        out = self.conv13(out,edge_index,selections,interps)
        out = self.relu(out)
        out = self.conv14(out,edge_index,selections,interps)
        out = self.relu(out)
        out = self.conv15(out,edge_index,selections,interps)
        out = self.relu(out)
        
        out = unpoolCluster(out,graph.clusters[1])
        edge_index = graph.edge_indexes[1]
        selections = graph.selections_list[1]
        interps = graph.interps_list[1] if hasattr(graph,'interps_list') else None

        out = self.conv16(out,edge_index,selections,interps)
        out = self.relu(out)
        out = self.conv17(out,edge_index,selections,interps)
        out = self.relu(out)
        
        out = unpoolCluster(out,graph.clusters[0])
        edge_index = graph.edge_indexes[0]
        selections = graph.selections_list[0]
        interps = graph.interps_list[0] if hasattr(graph,'interps_list') else None
        
        out = self.conv18(out,edge_index,selections,interps)
        out = self.relu(out)
        out = self.conv19(out,edge_index,selections,interps)

        return out

