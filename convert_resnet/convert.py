import torchvision
import torch
import openvino as ov

# OpenVINO 공식 문서를 참조해 torchvision 의 resnet50 모델을 OpenVINO 형식으로
# 변환시켜보자.
# 변환 후 ./ckpts/model.xml 로 저장하자.
# 참고: https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-pytorch.html

# ---start---

model = torchvision.models.resnet50(weights='DEFAULT')
ov_model = ov.convert_model(model,  example_input=torch.rand(1, 3, 224, 224))

ov.save_model(ov_model, './ckpts/model.xml')

# ---end---