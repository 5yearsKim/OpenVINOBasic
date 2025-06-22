from optimum.intel import OVModelForCausalLM

model_id = "microsoft/Phi-3-mini-4k-instruct"

model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)

# save the model after optimization
model.save_pretrained('./ckpts/phi3_8bit')

