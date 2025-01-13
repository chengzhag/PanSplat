import torch
from einops import rearrange

from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine

import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, norm_type='IN', **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        if norm_type == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        elif norm_type == 'BN':
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        return y

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class BackboneCascaded(torch.nn.Module):
    """docstring for BackboneCascaded."""

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_cross_attn=False,
        num_head=1,
    ):
        super(BackboneCascaded, self).__init__()
        self.feature_channels = feature_channels
        # Table 3: w/o cross-view attention
        self.no_cross_attn = no_cross_attn
        # Table B: w/ Epipolar Transformer

        self.encoder = FPNEncoder(feat_chs=feature_channels[::-1])
        self.decoder = FPNDecoder(feat_chs=feature_channels[::-1])

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels[0],
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.encoder(concat)
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(v)]
        for feature in features:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            for idx in range(v):
                features_list[idx].append(feature[:, idx])

        return features_list

    def decode_feature(self, features):
        b, v = features[0].shape[:2]
        features_list = []
        for feature in features:
            feature = rearrange(feature, "b v c h w -> (b v) c h w", b=b, v=v)
            features_list.append(feature)
        features_list = self.decoder(features_list[::-1])
        features = []
        for feature in features_list:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            features.append(feature)
        return features

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of features

        cur_features_list = [x[0] for x in features_list]

        # add position to features
        cur_features_list = feature_add_position_list(
            cur_features_list, attn_splits, self.feature_channels[0])

        # Transformer
        cur_features_list = self.transformer(
            cur_features_list, attn_num_splits=attn_splits)

        cur_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]
        cnn_features = []
        for idx in range(len(features_list[0])):
            cnn_features.append(torch.stack([x[idx] for x in features_list], dim=1))
        features = cnn_features.copy()
        features[0] = cur_features
        features = self.decode_feature(features)

        return {
            "trans_features": features,
            "cnn_features": cnn_features,
        }


class FPNEncoder(nn.Module):
    def __init__(self, feat_chs, norm_type='BN'):
        self.feat_chs = feat_chs
        super(FPNEncoder, self).__init__()
        self.conv00 = Conv2d(3, feat_chs[0], 7, 1, padding=3, norm_type=norm_type)
        self.conv01 = Conv2d(feat_chs[0], feat_chs[0], 5, 1, padding=2, norm_type=norm_type)

        for i, c in enumerate(feat_chs[1:]):
            setattr(self, f'downsample{i+1}', Conv2d(feat_chs[i], c, 5 if i < 2 else 3, 2, padding=2 if i < 2 else 1, norm_type=norm_type))
            setattr(self, f'conv{i+1}0', Conv2d(c, c, 3, 1, padding=1, norm_type=norm_type))
            setattr(self, f'conv{i+1}1', Conv2d(c, c, 3, 1, padding=1, norm_type=norm_type))

    def forward(self, x):
        x = self.conv00(x)
        x = self.conv01(x)
        output = [x]

        for i in range(1, len(self.feat_chs)):
            x = getattr(self, f'downsample{i}')(x)
            x = getattr(self, f'conv{i}0')(x)
            x = getattr(self, f'conv{i}1')(x)
            output.append(x)

        return output


class FPNDecoder(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoder, self).__init__()
        feat_chs = feat_chs[::-1]
        final_ch = feat_chs[0]
        self.out0 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[0], kernel_size=1), nn.BatchNorm2d(feat_chs[0]), Swish())

        for i, c in enumerate(feat_chs[1:]):
            setattr(self, f'inner{i+1}', nn.Conv2d(c, final_ch, 1))
            setattr(self, f'out{i+1}', nn.Sequential(nn.Conv2d(final_ch, c, kernel_size=3, padding=1), nn.BatchNorm2d(c), Swish()))

    def forward(self, xs):
        intra_feat = xs[-1]
        output = [self.out0(intra_feat)]

        for i in range(1, len(xs)):
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + getattr(self, f'inner{i}')(xs[-i-1])
            output.append(getattr(self, f'out{i}')(intra_feat))

        return output
