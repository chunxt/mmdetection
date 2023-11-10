from torchvision.models import resnet50
import torch
from nni.compression.pytorch.utils.counter import  count_flops_params
model = resnet50()
dummy_input = torch.randn(1, 3, 224, 224)
flops, params, results = count_flops_params(model, dummy_input)