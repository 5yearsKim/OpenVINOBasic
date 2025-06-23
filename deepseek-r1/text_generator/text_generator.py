import openvino_genai as ov_genai
import sys

from llm_config import convert_and_compress_model


class TextGenerator:
    # ---start---
    def __init__(self):
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



        self.pipe = ov_genai.LLMPipeline(str(model_dir), device)
        if "genai_chat_template" in model_configuration:
            self.pipe.get_tokenizer().set_chat_template(model_configuration["genai_chat_template"])

    
    def generate(self, prompt: str) -> str:
        def streamer(subword):
            print(subword, end="", flush=True)
            sys.stdout.flush()
            # Return flag corresponds whether generation should be stopped.
            # False means continue generation.
            return False

        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = 128

        result = self.pipe.generate(prompt, generation_config, streamer)
        return result
    
    # ---end---

