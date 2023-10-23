_base_ = [
    '/home/ry/DLtcx/exp_master/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/home/ry/DLtcx/exp_master/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/home/ry/DLtcx/exp_master/mmdetection/configs/_base_/default_runtime.py'
]
data_root = '/home/ry/DLry/mmdetection-dev-3.1/datasets/'


# model settings
model = dict(
    type='TOOD',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(1, 2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(type='Pretrained', checkpoint='/home/ry/DLtcx/checkpoint/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        prefix = 'backbone.')
        ),
    neck=dict(
        type='ACFPN',
        in_channels_list=[128, 256, 512, 1024],  #in_channels_list
        out_channels=128,
        # start_level=1,
        # add_extra_convs='on_output',
        # num_outs=5
        ),
    bbox_head=dict(
        type='TOODHead',
        num_classes=10,
        in_channels=128,
        stacked_convs=6,
        feat_channels=128,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
