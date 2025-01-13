from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from .resnet import *
from .mobilenet import *
from .layers import Conv3x3, ConvBlock, upsample, Cube2Equirec, Concat, BiProj, CEELayer
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms

class UniFuse(nn.Module):
    """ UniFuse Model: Resnet based Euqi Encoder and Cube Encoder + Euqi Decoder
    """
    def __init__(self, num_layers, equi_h, equi_w, pretrained=False, max_depth=10.0,
                 fusion_type="cee", se_in_fusion=True, mono_uncertainty=False, mono_lowres_pred=False):
        super(UniFuse, self).__init__()

        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h//2

        self.fusion_type = fusion_type
        self.se_in_fusion = se_in_fusion
        self.mono_uncertainty = mono_uncertainty
        self.mono_lowres_pred = mono_lowres_pred

        # encoder
        encoder = {2: mobilenet_v2,
                   18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,
                   152: resnet152}

        if num_layers not in encoder:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        self.equi_encoder = encoder[num_layers](pretrained)
        self.cube_encoder = encoder[num_layers](pretrained)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if num_layers < 18:
            self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        # decoder
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.equi_decoder = nn.ModuleDict()
        self.c2e = nn.ModuleDict()

        Fusion_dict = {"cat": Concat,
                       "biproj": BiProj,
                       "cee": CEELayer}
        FusionLayer = Fusion_dict[self.fusion_type]


        self.c2e["5"] = Cube2Equirec(self.cube_h // 32, self.equi_h // 32, self.equi_w // 32)

        self.equi_decoder["fusion_5"] = FusionLayer(self.num_ch_enc[4], SE=self.se_in_fusion)
        self.equi_decoder["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        self.c2e["4"] = Cube2Equirec(self.cube_h // 16, self.equi_h // 16, self.equi_w // 16)
        self.equi_decoder["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
        self.equi_decoder["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_decoder["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.c2e["3"] = Cube2Equirec(self.cube_h // 8, self.equi_h // 8, self.equi_w // 8)
        self.equi_decoder["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
        self.equi_decoder["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_decoder["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.c2e["2"] = Cube2Equirec(self.cube_h // 4, self.equi_h // 4, self.equi_w // 4)
        self.equi_decoder["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
        self.equi_decoder["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_decoder["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.c2e["1"] = Cube2Equirec(self.cube_h // 2, self.equi_h // 2, self.equi_w // 2)
        self.equi_decoder["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)
        self.equi_decoder["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_decoder["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
        self.equi_decoder["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
        self.equi_decoder["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)
        if self.mono_lowres_pred:
            self.lowres_depth_conv = None#Conv3x3(self.num_ch_dec[1], 1)

        self.sigmoid = nn.Sigmoid()
        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input_equi_image, input_cube_image, dnet=True, uncert_out=False):
        input_equi_image = self.normalize(input_equi_image)
        input_cube_image = self.normalize(input_cube_image)

        # euqi image encoding

        if self.num_layers < 18:
            equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4 \
                = self.equi_encoder(input_equi_image)
        else:
            x = self.equi_encoder.conv1(input_equi_image)
            x = self.equi_encoder.relu(self.equi_encoder.bn1(x))
            equi_enc_feat0 = x

            x = self.equi_encoder.maxpool(x)
            equi_enc_feat1 = self.equi_encoder.layer1(x)
            equi_enc_feat2 = self.equi_encoder.layer2(equi_enc_feat1)
            equi_enc_feat3 = self.equi_encoder.layer3(equi_enc_feat2)
            equi_enc_feat4 = self.equi_encoder.layer4(equi_enc_feat3)


        # cube image encoding
        cube_inputs = torch.cat(torch.split(input_cube_image, self.cube_h, dim=-1), dim=0)

        if self.num_layers < 18:
            cube_enc_feat0, cube_enc_feat1, cube_enc_feat2, cube_enc_feat3, cube_enc_feat4 \
                = self.cube_encoder(cube_inputs)
        else:

            x = self.cube_encoder.conv1(cube_inputs)
            x = self.cube_encoder.relu(self.cube_encoder.bn1(x))
            cube_enc_feat0 = x

            x = self.cube_encoder.maxpool(x)

            cube_enc_feat1 = self.cube_encoder.layer1(x)
            cube_enc_feat2 = self.cube_encoder.layer2(cube_enc_feat1)
            cube_enc_feat3 = self.cube_encoder.layer3(cube_enc_feat2)
            cube_enc_feat4 = self.cube_encoder.layer4(cube_enc_feat3)


        # euqi image decoding fused with cubemap features
        outputs = {}

        cube_enc_feat4 = torch.cat(torch.split(cube_enc_feat4, input_equi_image.shape[0], dim=0), dim=-1)

        c2e_enc_feat4 = self.c2e["5"](cube_enc_feat4)
        # import ipdb;ipdb.set_trace()
        fused_feat4 = self.equi_decoder["fusion_5"](equi_enc_feat4, c2e_enc_feat4)

        equi_x = upsample(self.equi_decoder["upconv_5"](fused_feat4))
        cube_enc_feat3 = torch.cat(torch.split(cube_enc_feat3, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat3 = self.c2e["4"](cube_enc_feat3)
        fused_feat3 = self.equi_decoder["fusion_4"](equi_enc_feat3, c2e_enc_feat3)
        equi_x = torch.cat([equi_x, fused_feat3], 1)
        equi_x = self.equi_decoder["deconv_4"](equi_x)
        equi_x = upsample(self.equi_decoder["upconv_4"](equi_x))

        cube_enc_feat2 = torch.cat(torch.split(cube_enc_feat2, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat2 = self.c2e["3"](cube_enc_feat2)
        fused_feat2 = self.equi_decoder["fusion_3"](equi_enc_feat2, c2e_enc_feat2)
        equi_x = torch.cat([equi_x, fused_feat2], 1)
        equi_x = self.equi_decoder["deconv_3"](equi_x)
        equi_x = upsample(self.equi_decoder["upconv_3"](equi_x))

        cube_enc_feat1 = torch.cat(torch.split(cube_enc_feat1, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat1 = self.c2e["2"](cube_enc_feat1)
        fused_feat1 = self.equi_decoder["fusion_2"](equi_enc_feat1, c2e_enc_feat1)
        equi_x = torch.cat([equi_x, fused_feat1], 1)
        equi_x = self.equi_decoder["deconv_2"](equi_x)
        equi_x = upsample(self.equi_decoder["upconv_2"](equi_x))

        cube_enc_feat0 = torch.cat(torch.split(cube_enc_feat0, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat0 = self.c2e["1"](cube_enc_feat0)
        fused_feat0 = self.equi_decoder["fusion_1"](equi_enc_feat0, c2e_enc_feat0)

        equi_x = torch.cat([equi_x, fused_feat0], 1)
        equi_x = self.equi_decoder["deconv_1"](equi_x)

        mono_feat = equi_x
        if self.mono_lowres_pred:
            lowres_pred = self.lowres_depth_conv(equi_x)
            outputs["lowres_pred"] = self.activation_G(lowres_pred)

        equi_x = upsample(self.equi_decoder["upconv_1"](equi_x))
        equi_x = self.equi_decoder["deconv_0"](equi_x)
        if self.mono_uncertainty:
            equi_outs = self.equi_decoder["depthconv_0"](equi_x)
            # equi_sigma = self.uncertainty_conv(equi_x)
            # equi_outs = torch.cat([equi_depth, equi_sigma], dim=1)
            outputs["pred"] = self.activation_G(equi_outs)
            # outputs["pred"][:, 0, ...] = self.max_depth * self.sigmoid(outputs["pred"][:, 0, ...])
            # outputs["pred"][:, 1, ...] = outputs["pred"][:, 1, ...]
        else:
            equi_depth = self.equi_decoder["depthconv_0"](equi_x)
            outputs["pred_depth"] = self.max_depth * self.sigmoid(equi_depth)

        if uncert_out:
            outputs.update({
                "fused_feat4": fused_feat4,
                "fused_feat3": fused_feat3,
                "fused_feat2": fused_feat2,
                "fused_feat1": fused_feat1,
                "fused_feat0": fused_feat0,                
            })
            # return outputs #, mono_feat, fused_feat4, fused_feat3, fused_feat2, fused_feat1, fused_feat0
        if dnet:
            outputs["mono_feat"] = mono_feat
        # if dnet:
        #     return outputs, mono_feat
        return outputs
    # outputs["pred"] = self.activation_G(equi_outs)
    def activation_G(self, out):
        mu, var = torch.split(out, 1, dim=1)
        var = F.elu(var) + 1.0 + 1e-10
        out = torch.cat([mu, var], dim=1)  # (N, 2, H, W)
        return out
