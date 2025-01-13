# from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .layers import Conv3x3_wrap, Conv3x3
#
use_wrap_padding=True
class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding, dilation):
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, dilation)
    def forward(self,x):

        if use_wrap_padding:
            x = F.pad(x, (0, 0, self.padding, self.padding), mode = 'constant')
            x = F.pad(x, ( self.padding, self.padding, 0, 0), mode = 'circular')
            return F.max_pool2d(x, self.kernel_size, self.stride,
                            0, self.dilation, self.ceil_mode,
                            self.return_indices)
        else:
            return F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)


# Conv2d with weight standardization
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        # weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
        #                           keepdim=True).mean(dim=3, keepdim=True)
        # weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        # weight = weight / std.expand_as(weight)
        if use_wrap_padding:
            if isinstance(self.padding,int):
                x = F.pad(x, (0, 0, self.padding, self.padding), mode = 'constant')
                x = F.pad(x, ( self.padding, self.padding, 0, 0), mode = 'circular')
            else:#tuple
                pad_h = self.padding[0]
                pad_w = self.padding[1]
                x = F.pad(x, (0, 0, pad_h, pad_h), mode = 'constant')
                x = F.pad(x, (pad_w, pad_w, 0, 0), mode = 'circular')

            return F.conv2d(x, weight, self.bias, self.stride,
                            0, self.dilation, self.groups)

        else:
            return F.conv2d(x, weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)




# ref1: https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736/3
# ref2: OmniFusion
def get_new_layer(m, target_attr, bias=False):
    new_layer = Conv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride, \
        padding=m.padding, dilation=m.dilation, bias=bias)               
    # import pdb;pdb.set_trace()
    # new_layer.cuda()
    return new_layer

def get_new_conv3x3(m, target_attr, bias=False):
    new_layer = Conv3x3_wrap(m.in_channels, m.out_channels, bias=m.bias)
    return new_layer

# nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
def get_new_maxpool(m, target_attr):
    new_maxpool=MaxPool2d(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation)
    return new_maxpool

    
def setup_conv(target_attr):
    
    m = copy.deepcopy(target_attr)
    # import pdb;pdb.set_trace()
    if m.bias is not None:
        new_layer = get_new_layer(m, target_attr, bias=True) 
        new_layer.bias.data.copy_(m.bias.data)

    else:
        new_layer = get_new_layer(m, target_attr, bias=False) 
    new_layer.weight.data.copy_(m.weight.data)#.unsqueeze(-1))

    return new_layer

def setup_conv3x3(target_attr):
    
    m = copy.deepcopy(target_attr)
    # import pdb;pdb.set_trace()
    if m.bias is not None:
        new_layer = get_new_conv3x3(m, target_attr, bias=True) 
        new_layer.conv.bias.data.copy_(m.conv.bias.data)
    # import ipdb;ipdb.set_trace()
    # if hasattr(m, 'bias'):
    #     new_layer = get_new_conv3x3(m, target_attr, bias=m.bias) 
    # else:
    else:
        new_layer = get_new_conv3x3(m, target_attr, bias=False) 
    #conv3x3  does not have weight to load from pretrained.
    new_layer.conv.weight.data.copy_(m.conv.weight.data)#.unsqueeze(-1))
    return new_layer

def setup_maxpool(target_attr):
    m = copy.deepcopy(target_attr)    
    new_layer = get_new_maxpool(m, target_attr) 
    # new_layer.weight.data.copy_(m.weight.data)#.unsqueeze(-1))
    return new_layer

    # first level of current layer or model contains a batch norm --> replacing.
    # model._modules[name] = copy.deepcopy(new_layer)                

# def setup_bn(target_attr):
#     m = copy.deepcopy(target_attr)
#     new_layer = nn.BatchNorm3d(m.num_features)
#     new_layer.weight.data.copy_(m.weight.data)
#     new_layer.bias.data.copy_(m.bias.data)
#     new_layer.running_mean.data.copy_(m.running_mean.data)
#     new_layer.running_var.data.copy_(m.running_var.data)
#     return new_layer

def replace_conv(module, name):
    if type(module) == torch.nn.Conv2d or isinstance(module, torch.nn.Conv2d):
        new_layer = setup_conv(module)   
        module = new_layer
        # import pdb;pdb.set_trace()
        return new_layer
        
    
    for n, ch in module.named_children():
        ch = replace_conv(ch, n)
        setattr(module, n, ch)
    return module




def replace_maxpool(module, name):
    if type(module) == torch.nn.MaxPool2d or isinstance(module, torch.nn.MaxPool2d):
        new_layer = setup_maxpool(module)   
        module = new_layer
        # import pdb;pdb.set_trace()
        return new_layer    
    for n, ch in module.named_children():
        ch = replace_maxpool(ch, n)
        setattr(module, n, ch)
    return module

def replace_conv3x3(module, name):
    if type(module) == Conv3x3 or isinstance(module, Conv3x3):
        new_layer = setup_conv3x3(module)
        module = new_layer
        return new_layer

    for n, ch in module.named_children():
        ch = replace_conv3x3(ch, n)
        setattr(module, n, ch)
    return module

def erp_convert(model):
    #b_list = [getattr(self.conv_stem, attr_str) for attr_str in dir(self.conv_stem)] 
    
    model = replace_conv(model, "net")
    model = replace_maxpool(model, 'net')
    model = replace_conv3x3(model, 'net')
    return model
