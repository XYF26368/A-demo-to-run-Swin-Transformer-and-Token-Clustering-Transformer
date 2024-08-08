_base_ = [
    './tcf-base-patch4-nonpre-ade20k-512x512.py'
]
checkpoint_file = "/home/work_dirs/tcformer-base-patch4-nonpre-ade20k-512x512/iter_10000.pth"  
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[96,192,384,768],#[64, 128, 256, 512]
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[96,192,384,768], num_classes=150),#128,256,512,1024
    auxiliary_head=dict(in_channels=384, num_classes=150))#512