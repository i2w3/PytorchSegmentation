'''
测试 torch.compile 对模型推理速度的影响
'''
import time

import torch
import segmentation_models_pytorch as smp


def test_time(model, dummy_input, iterations=100):
    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)
    # Timing
    start_time = time.time()
    for _ in range(iterations):
        _ = model(dummy_input)
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    dummy_input = torch.randn(8, 3, 512, 512).cuda()
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    ).cuda()

    # Measure performance
    print(f"[EAGER] Time without torch.compile: {test_time(model, dummy_input):.4f} seconds")

    model_compiled = torch.compile(model)
    print(f"[COMPILE] Time with torch.compile: {test_time(model_compiled, dummy_input):.4f} seconds")