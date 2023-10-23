from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '/home/ry/DLtcx/exp_master/mmdetection/work_dirs/tood_csp_fpn_1x_coco_cc58.2/tood_csp_fpn_1x_coco.py'
checkpoint_file = '/home/ry/DLtcx/exp_master/mmdetection/work_dirs/tood_csp_fpn_1x_coco_cc58.2/epoch_10.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/ry/DLry/mmdetection-dev-3.1/datasets/VisDrone2019-DET/VisDrone2019-DET-val/images/0000001_08414_d_0000013.jpg'
# '/public/home/cxj123456/DLzyk/dataset/data/VisDrone2019-DET-val/images/0000330_04201_d_0000821.jpg'
#  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# save the visualization results to image files
show_result_pyplot(model, img, result)