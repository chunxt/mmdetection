_base_ = [
    '/home/ry/DLtcx/exp_master/mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '/home/ry/DLtcx/exp_master/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/home/ry/DLtcx/exp_master/mmdetection/configs/_base_/schedules/schedule_1x.py', '/home/ry/DLtcx/exp_master/mmdetection/configs/_base_/default_runtime.py',
    '/home/ry/DLtcx/exp_master/mmdetection/configs/retinanet/retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
