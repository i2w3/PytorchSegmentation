'''
测试 segmentation_models_pytorch 中各种模型能否被 torch.compile 成功编译运行
'''

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_float32_matmul_precision('high')
torch._dynamo.config.recompile_limit = 100
torch._dynamo.config.accumulated_recompile_limit = 100
import segmentation_models_pytorch as smp


if __name__ == "__main__":
    # support N * 32
    model_list = ["Unet", "UnetPlusPlus", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "Linknet", "MAnet", "PAN", "UPerNet", "Segformer"]
    dummy_input = torch.randn(1, 3, 512, 512).cuda()
    for model_name in model_list:
        model = getattr(smp, model_name)(in_channels=3, classes=5).cuda().eval()
        model = torch.compile(model)
        output = model(dummy_input)
        print(f"[SUCCESS] Model {model_name} - {model.name} compiled and ran successfully.")
    
    # only support 224 * 224
    model_list = ["DPT"]
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    for model_name in model_list:
        model = getattr(smp, model_name)(in_channels=3, classes=5).cuda().eval()
        model = torch.compile(model)
        output = model(dummy_input)
        print(f"[SUCCESS] Model {model_name} - {model.name} compiled and ran successfully.")