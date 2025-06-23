import sys
from pathlib import Path

import openvino_genai as ov_genai
import requests

from llm_config import compare_model_size, convert_and_compress_model

if not Path("llm_config.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py"
    )
    open("llm_config.py", "w").write(r.text)

if not Path("notebook_utils.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"
    )
    open("notebook_utils.py", "w").write(r.text)

if not Path("genai_helper.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/genai_helper.py"
    )
    open("genai_helper.py", "w").write(r.text)

model_id = "DeepSeek-R1-Distill-Qwen-1.5B"
compression_variant = "INT4"
device = "CPU"

model_configuration = {
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
    "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
    "stop_strings": [
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "</User|>",
        "<|User|>",
        "<|end_of_sentence|>",
        "</｜",
    ],
}


model_dir = convert_and_compress_model(
    model_id, model_configuration, compression_variant, use_preconverted=True
)


compare_model_size(model_dir)


print(f"Loading model from {model_dir}\n")


pipe = ov_genai.LLMPipeline(str(model_dir), device)
if "genai_chat_template" in model_configuration:
    pipe.get_tokenizer().set_chat_template(model_configuration["genai_chat_template"])

generation_config = ov_genai.GenerationConfig()
generation_config.max_new_tokens = 128


def streamer(subword):
    print(subword, end="", flush=True)
    sys.stdout.flush()
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False


input_prompt = "What is OpenVINO?"
print(f"Input text: {input_prompt}")
result = pipe.generate(input_prompt, generation_config, streamer)
