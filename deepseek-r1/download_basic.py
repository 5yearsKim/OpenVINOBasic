from pathlib import Path
import requests

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

