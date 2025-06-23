from optimum.intel import OVModelForCausalLM

# microsoft/Phi-3-mini-4k-instruct 허깅페이스 모델을 OpenVINO 모델로
# 변환하는 코드를 완성해보자.
# 변환 후 ./ckpts/phi3_8bit 에 저장하자.
# 다음 공식문서를 참고하자.
# https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html
# ---start---

model_id = "microsoft/Phi-3-mini-4k-instruct"

model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)

# save the model after optimization
model.save_pretrained('./ckpts/phi3_8bit')

# ---end---

