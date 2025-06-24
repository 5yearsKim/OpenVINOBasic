import time
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.streamers import TextStreamer

model_id = 'microsoft/Phi-3-mini-4k-instruct'
model_path = './ckpts/phi3_8bit'


# original_inference.py 코드를 참조해서 openvino 변환 모델로 똑같이
# inference 성능을 측정하는 코드를 작성해보자.

# ---start---
# Fill your code
# ---end---