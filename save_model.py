import torchvision
import torch
import onnx
net=torchvision.models.resnet18(pretrained=True)
net.eval()

# 导出pt模型
torch.save(net, 'resnet18.pt')

# 导出onnx
inputs = torch.ones([1, 3, 224, 224])
torch.onnx.export(net, inputs, 'resnet18.onnx', input_names=["input0"],
                  output_names=["output0"], verbose=False)

# Checks
model_onnx = onnx.load('resnet18.onnx')  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx mode