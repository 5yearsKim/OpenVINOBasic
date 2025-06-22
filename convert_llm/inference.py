import time
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.streamers import TextStreamer


model_id = 'microsoft/Phi-3-mini-4k-instruct'
model_path = './ckpts/phi3_8bit'

model = OVModelForCausalLM.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


prompt = "I am a korean, and I love "

inputs = tokenizer(prompt, return_tensors="pt")

start_t = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=50, streamer=streamer)
end_t = time.perf_counter()

generated_tokens = outputs.shape[-1] - inputs['input_ids'].shape[-1]
elapsed_time = end_t - start_t
tps = generated_tokens / elapsed_time



# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"\nGenerated {generated_tokens} tokens in {elapsed_time:.2f}s → {tps:.2f} tokens/sec")
print("finished")

