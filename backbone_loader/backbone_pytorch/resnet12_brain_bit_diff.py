
import torch
import torch.nn as nn
import random # for manifold mixup
from functools import partial

import brevitas.nn as qnn  
from brevitas.quant import Int8WeightPerTensorFixedPoint
from brevitas.quant import Int8ActPerTensorFixedPoint
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import RestrictValueType

class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFixedPoint):  
    scaling_min_val = 2e-16
    bit_width = None

class CommonIntActPerTensorQuant(Int8ActPerTensorFixedPoint):
    scaling_min_val = 2e-16
    bit_width = None

class QuantConvBN2d(qnn.QuantConv2d):  
    def __init__(self, in_f, out_f, m, n, ma, na, kernel_size=3, stride=1, padding=1, groups=1, outRelu=True, leaky=False):
        super(QuantConvBN2d, self).__init__(in_f, out_f, kernel_size = kernel_size)

        self.relu_test = qnn.QuantReLU(act_quant = CommonIntActPerTensorQuant, bit_width = m+n)    
        self.conv = qnn.QuantConv2d(in_f, out_f, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups, bias = False, #fixed_quantization
                                    weight_quant = CommonIntWeightPerTensorQuant, 
                                    weight_bit_width = m+n,  
                                    max_val= 2.0**(m-1)- 2.0**(-n), 
                                    min_val = -2.0 ** (m-1), 
                                    weight_restrict_scaling_type = RestrictValueType.POWER_OF_TWO,
                                    weight_scaling_impl_type = ScalingImplType.CONST,                                     
                                    weight_scaling_const = 2.0**(m-1), 
                                    )  
        self.bn = nn.BatchNorm2d(out_f) 
        self.relu = qnn.QuantReLU(act_quant = CommonIntActPerTensorQuant, 
                                        bit_width=ma+na, 
                                        max_val= 2.0**(ma-1)- 2.0**(-na),
                                        min_val = -2.0 ** (ma-1),
                                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                        scaling_impl_type=ScalingImplType.CONST,
                                        act_scaling_const = 2.0**(ma-1),
                                        signed = False,
                                        )
        self.relu_leaky = qnn.QuantIdentity(act_quant = CommonIntActPerTensorQuant, 
                                        bit_width=ma+na, 
                                        max_val= 2.0**(ma-1)- 2.0**(-na),
                                        min_val = -2.0 ** (ma-1),
                                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                        scaling_impl_type=ScalingImplType.CONST,
                                        act_scaling_const = 2.0**(ma-1), 
                                        ) 
        self.outRelu = outRelu
        self.leaky = leaky
        if leaky:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, lbda = None, perm = None):
        y = self.bn(self.conv(x))

        if lbda is not None:
            y = lbda * y + (1 - lbda) * y[perm]

        if self.outRelu:
            if not self.leaky:
                y = self.relu(y)
                return y   
            else:
                y = nn.functional.leaky_relu(y, negative_slope = 0.1)
                y = self.relu_leaky(y)
                return y
        else:
            return y 

class BasicBlockRN12(nn.Module):
    def __init__(self, in_f, out_f, bits, use_strides, block_name, leaky=False):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = QuantConvBN2d(in_f, out_f, outRelu=True, **bits['conv1'])
        self.conv2 = QuantConvBN2d(out_f, out_f, outRelu=True, **bits['conv2'])

        self.block_name = block_name
        if block_name == 'block3': #MARK: ここの記述確認
            ma, na =2,1
        else:
            ma, na = 2,1

        self.relu = qnn.QuantReLU(act_quant = CommonIntActPerTensorQuant, 
                                bit_width=ma+na, 
                                max_val= 2.0**(ma-1)- 2.0**(-na),
                                min_val = -2.0 ** (ma-1),
                                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                scaling_impl_type=ScalingImplType.CONST,
                                act_scaling_const = 2.0**(ma-1),
                                signed = False,
                                ) 
        self.relu_leaky = qnn.QuantIdentity(act_quant = CommonIntActPerTensorQuant, 
                                        bit_width=ma+na, 
                                        max_val= 2.0**(ma-1)- 2.0**(-na),
                                        min_val = -2.0 ** (ma-1),
                                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                        scaling_impl_type=ScalingImplType.CONST,
                                        act_scaling_const = 2.0**(ma-1) #??
                                        ) 

        if use_strides: 
            self.conv3 = QuantConvBN2d(out_f, out_f, stride=2, **bits['conv3'])
            self.sc = QuantConvBN2d(in_f, out_f , kernel_size = 1, padding = 0,stride=2, **bits['sc'])

        else:
            self.conv3 = QuantConvBN2d(out_f, out_f, **bits['conv3'])
            self.sc = QuantConvBN2d(in_f, out_f, kernel_size = 1, padding = 0, **bits['conv3'])

        self.leaky = leaky


    def forward(self, x, lbda = None, perm = None):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y += self.sc(x)
        
        if lbda is not None:
            y = lbda * y + (1 - lbda) * y[perm]

        if self.leaky:
            y = torch.nn.functional.leaky_relu(y, negative_slope = 0.1)  
            y = self.relu_leaky(y)
            return y

        else:
            y = self.relu(y)
            return y    

class ResNet9(nn.Module):
    def __init__(self, feature_maps, use_strides=True):
        super(ResNet9, self).__init__()

    ## Define bit configuration for each block and layer
        bit_config = {
        'block1': {
        'conv1': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'conv2': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'conv3': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'sc':    {'m': 1, 'n': 5, 'ma': 2, 'na': 1}
    },
    'block2': {
        'conv1': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'conv2': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'conv3': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'sc':    {'m': 1, 'n': 5, 'ma': 2, 'na': 1}
    },
    'block3': {
        'conv1': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'conv2': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'conv3': {'m': 1, 'n': 5, 'ma': 2, 'na': 1},
        'sc':    {'m': 1, 'n': 5, 'ma': 2, 'na': 1}
    } 
        }
    ## Define bit configuration for each block and layer

        self.block1 = BasicBlockRN12(3, feature_maps, bit_config['block1'], use_strides=use_strides, block_name='block1')
        self.block2 = BasicBlockRN12(feature_maps, int(2.5 * feature_maps), bit_config['block2'], use_strides=use_strides, block_name='block2')
        self.block3 = BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps, bit_config['block3'], use_strides=use_strides, block_name='block3')
        self.mp = nn.Identity() if use_strides else nn.MaxPool2d(2)

    def forward(self, x, mixup = None, lbda = None, perm = None):

        mixup_layer = -1
        if mixup == "mixup":
            mixup_layer = 0
        elif mixup == "manifold mixup":
            mixup_layer = random.randint(0, 4)

        if mixup_layer == 0:
            x = lbda * x + (1 - lbda) * x[perm]
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        if mixup_layer == 1:
            y = self.mp(self.block1(x, lbda, perm))
        else:
            y = self.mp(self.block1(x))

        if mixup_layer == 2:
            y = self.mp(self.block2(y, lbda, perm))
        else:
            y = self.mp(self.block2(y))

        if mixup_layer == 3:
            y = self.mp(self.block3(y, lbda, perm))
        else:
            y = self.mp(self.block3(y))

        y = y.mean(dim = list(range(2, len(y.shape))))
        return y