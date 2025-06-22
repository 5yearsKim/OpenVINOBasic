import torchvision
import torch
import openvino as ov

model = torchvision.models.resnet50(weights='DEFAULT')
ov_model = ov.convert_model(model,  example_input=torch.rand(1, 3, 224, 224))

ov.save_model(ov_model, './ckpts/model.xml')