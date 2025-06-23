# 🧠 OpenVINO ResNet 변환 및 추론

 PyTorch의 ResNet 모델을 OpenVINO 형식으로 변환하고, 양자화를 적용한 후, 추론 성능을 측정

## 📦 환경 설정

체크포인트, 데이터 디렉토리 만들기
```sh
mkdir ckpts/ data/
```

가상환경 세팅
```sh
python3 -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🔄 모델 변환 및 양자화

### 1. ResNet 모델을 OpenVINO IR 형식으로 변환

```sh
python convert.py
```

> 기본적으로 PyTorch의 `torchvision.models.resnet50(weights="DEFAULT")` 모델을 사용하여 `ckpts/model.xml`로 저장

### 2. 양자화 적용 (옵션)

---
python quantize.py
---

> 양자화된 모델은 `ckpts/quantized_model.xml`로 저장됨.

---

## 🚀 추론 및 성능 측정

```sh
python inference.py
```

- PyTorch, OpenVINO, 양자화 모델 각각에 대해 추론을 실행하고 성능을 비교
- 출력: 추론 시간, 예측 결과 등

--- 

## 참조
- [Conventional Convert from PyTorch](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-pytorch.html)
- [OpenVINO Quantization](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html)

