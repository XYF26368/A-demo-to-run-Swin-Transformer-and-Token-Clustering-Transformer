import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from mmseg.registry import MODELS
from functools import partial
import math
import sys
from mmengine.utils import to_2tuple
from mmcv.cnn import build_norm_layer
sys.path.append("/home/TCFormer_n/mmseg/models/TCFormerBase")
# sys.path.append("/home/TCFormer_n/mmseg/models/utils")
from mmcv.cnn import build_conv_layer
from tcformer_layers import (
    TCBlock, OverlapPatchEmbed, CTM, Block
)

from tcformer_utils import (
    token_downup, token2map, get_root_logger, load_checkpoint
)
from mmcv.cnn import ConvModule
import warnings
import torch.nn.functional as F
from mmengine.logging import print_log
# from embed import PatchEmbed
# from transformer_utils import trunc_normal_
from mmengine.model.weight_init import trunc_normal_, constant_init, trunc_normal_init
from collections import OrderedDict
from mmengine.runner import CheckpointLoader



class MTA(BaseModule):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 out_channels= 96, # 128 [96, 192, 384, 768] [128,128,128,128]
                 num_outs=1,
                 start_level=0,
                 end_level=-1,
                 num_heads=[2, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4],
                 sr_ratios=[8, 4, 2, 1],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 no_norm_on_lateral=False,
                 conv_cfg=dict(type = 'Conv2d'),
                 norm_cfg=None,
                 act_cfg=dict(type='GELU'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 use_sr_layer=True,
                 ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.mlp_ratios = mlp_ratios

        self.start_level = start_level
        if end_level == -1:
            end_level = len(in_channels) - 1
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        self.norm = []
        for i in range(self.start_level, self.end_level + 1):
            l_conv = ConvModule(
                in_channels[i],
                self.out_channels*(2**i),# 输出通道数依次翻倍
                1,
                conv_cfg=conv_cfg,
                norm_cfg= None, #norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.norm.append(build_norm_layer(norm_cfg, self.out_channels*(2**i))[1])
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.end_level):
            merge_block = TCBlock(
                dim=self.out_channels*(2**i), num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], use_sr_layer=use_sr_layer,
            )
            self.merge_blocks.append(merge_block)

        # add extra conv layers (e.g., RetinaNet)
        self.relu_before_extra_convs = relu_before_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
            self.add_extra_convs = add_extra_convs
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        else:
            self.add_extra_convs = add_extra_convs

        self.extra_convs = nn.ModuleList()
        extra_levels = num_outs - (self.end_level + 1 - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.end_level]
                else:
                    _out_channels = [self.out_channels*(2**j) for j in range(self.start_level, self.end_level + 1)]
                    in_channels = _out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    _out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.extra_convs.append(extra_fpn_conv)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build lateral tokens
        input_dicts = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            tmp = inputs[i + self.start_level].copy()
            # print(tmp['x'].unsqueeze(2).permute(0, 3, 1, 2).shape)
            # tmp['x'] = tmp['x'].unsqueeze(2).permute(0, 3, 1, 2)
            # shape = tmp['x'].shape
            # tmp['x'] = tmp['x'].reshape(shape[2], -1)
            # tmp['x'] = lateral_conv(tmp['x']).view(shape).permute(0, 2, 3, 1).squeeze(2) 
            tmp['x'] = lateral_conv(tmp['x'].unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2) # 执行卷积操作 通道数从输入通道到输出通道
            # tmp['x'] = self.norm[i](lateral_conv(tmp['x'].unsqueeze(2).permute(0, 3, 1, 2))).permute(0, 2, 3, 1).squeeze(2) # 执行卷积操作 通道数从输入通道到输出通道
            input_dicts.append(tmp)

        # merge from high level to low level
        for i in range(len(input_dicts) - 2, -1, -1):# i取值从len(input_dicts) - 2到0
            # print(input_dicts[i]['map_size'])
            if token_downup(input_dicts[i], input_dicts[i + 1]).shape[2] > input_dicts[i]['x'].shape[2]:
                input_dicts[i]['x'] = input_dicts[i]['x'] + token_downup(input_dicts[i], input_dicts[i + 1]).narrow(2, 0, input_dicts[i]['x'].shape[2])
            else:
                input_dicts[i]['x'] = input_dicts[i]['x'] + token_downup(input_dicts[i], input_dicts[i + 1]) # 根据网络结构设计 特征维度不大于只可能等于 所以直接相加就行
            input_dicts[i] = self.merge_blocks[i](input_dicts[i])
        # transform to feature map
        outs = [token2map(token_dict) for token_dict in input_dicts]
        # for i in outs:
        #     print(i.shape)
        # part 2: add extra levels
        used_backbone_levels = len(outs)
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))

            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    tmp = inputs[self.end_level]
                    extra_source = token2map(tmp)
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(self.extra_convs[0](extra_source))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.extra_convs[i](F.gelu(outs[-1])))# 原本用的relu
                    else:
                        outs.append(self.extra_convs[i](outs[-1]))
        return outs



# TCFormer for segmentation in ADE20K
# 加一个 head 用来转化为分割任务
@MODELS.register_module()
class TCFormerSeg(BaseModule):
    def __init__(
            self, pretrain_img_size=224, in_channnels=3, in_channels_mta=[64, 128, 320, 512], 
            out_channels_mta= 128, embed_dims=[64, 128, 320, 512], patch_size=4,stride = 4,
            num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            num_stages=4, pretrained=None, frozen_stages = -1, 
            k=5, sample_ratios=[0.25, 0.25, 0.25],
            return_map=False, conv_cfg=dict(type = 'Conv2d'),
            norm_cfg=None,
            act_cfg=dict(type='GELU'),
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform'),
            **kwargs
    ):
        super().__init__(init_cfg=init_cfg)

        # 初始化参数
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.stride = stride,
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channnels
        self.in_channels_mta = in_channels_mta
        self.out_channels_mta= out_channels_mta
        self.k = k
        self.frozen_stages = frozen_stages
        self.mta_block = MTA(
            in_channels=self.in_channels_mta,
            out_channels= self.out_channels_mta, # 128 [96, 192, 384, 728] [128,128,128,128]
            num_outs=1,
            start_level=0,
            end_level=-1,
            num_heads=[2, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
            sr_ratios=[8, 4, 2, 1],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            no_norm_on_lateral=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform', checkpoint = None),
            add_extra_convs=False,
            extra_convs_on_inputs=True,
            relu_before_extra_convs=False,
            use_sr_layer=True,
        )

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')


        # 根据depths计算每一层的drop path率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # 第一个阶段，使用标准的transformer blocks
        # 这里的取尺寸必须采用向下取整算法才能保证跟swin的一样
        for i in range(1):
            patch_embed = OverlapPatchEmbed(
                img_size=pretrain_img_size,
                patch_size=self.patch_size,
                stride=self.stride,
                in_channnels=self.in_channels,
                embed_dim=embed_dims[i]
            )
            

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # 第二到第四阶段，使用TCBlock进行动态token处理
        for i in range(1, num_stages):
            ctm = CTM(sample_ratios[i-1], embed_dims[i-1], embed_dims[i], k)

            block = nn.ModuleList([TCBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"ctm{i}", ctm)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # 应用权重初始化
        self.apply(self._init_weights)
        self.init_weights(pretrained)
    
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def forward_features(self, x):
        outs = []

        i = 0
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)

        # 初始化token字典
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())
        # 第二到第四阶段
        for i in range(1, self.num_stages):
            ctm = getattr(self, f"ctm{i}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            token_dict = ctm(token_dict)  # 下采样
            for j, blk in enumerate(block):
                token_dict = blk(token_dict)

            token_dict['x'] = norm(token_dict['x'])
            outs.append(token_dict)
        if self.return_map:
            outs = [token2map(token_dict) for token_dict in outs]

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        x_out = self.mta_block(x)
        return x_out

if __name__ == "__main__":
    class tcformer(TCFormerSeg):
        def __init__(self, **kwargs):
            super().__init__(
                pretrain_img_size=512, 
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            init_cfg=dict(type='Pretrained', checkpoint=None),
            **kwargs)

    former = TCFormerSeg(pretrain_img_size=512,
        embed_dims=[64, 128, 256, 512],
        patch_size=4,
        mlp_ratio=4,
        depths=[3, 4, 6, 3],
        num_heads=[1, 2, 4, 8],
        in_channels_mta=[64, 128, 256, 512], 
        out_channels_mta= 128,
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        init_cfg=dict(type='Pretrained', checkpoint=None),
        act_cfg=dict(type='GELU'),
        norm_cfg=None)

    X = torch.randn(1,3,512,512)
    y = former(X)
    print(len(y))
    for i in range(len(y)):
        print(y[i].shape)


    # python tcformer.py

