import io
import time
from typing import Tuple, Any

import requests
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
import openvino as ov


def download_images(urls: list[str]) -> list[Image.Image]:
    """Download images from URLs and return PIL Image list."""
    imgs: list[Image.Image] = []
    for url in urls:
        resp = requests.get(url)
        resp.raise_for_status()
        imgs.append(Image.open(io.BytesIO(resp.content)).convert("RGB"))
    return imgs


def prepare_batch(img_list: list[Image.Image],
                  weights: ResNet50_Weights = ResNet50_Weights.DEFAULT) -> torch.Tensor:
    """Preprocess a PIL image into a batch tensor for the model."""
    preprocess = weights.transforms()
    return [preprocess(item).unsqueeze(0) for item in img_list]


def inference(model: Any, batch: torch.Tensor) -> Tuple[str, float]:
    """Run model inference and return (category_name, score)."""
    with torch.no_grad():
        output = model(batch)
        # Handle OpenVINO or PyTorch outputs
        if isinstance(output, torch.Tensor):
            logits = output.squeeze(0)
        else:
            # OpenVINO returns numpy arrays
            logits = torch.from_numpy(output[0]).squeeze(0)
        probs = logits.softmax(0)
    idx = int(probs.argmax())
    score = float(probs[idx])
    label = ResNet50_Weights.DEFAULT.meta["categories"][idx]
    return label, score


if __name__ == "__main__":
    # URLs for sample images

    IMG_URLS = [
        'https://picsum.photos/id/2/200/300', # laptop
        'https://picsum.photos/id/237/200/300', # dog
        'https://picsum.photos/id/3/200/300',
        'https://picsum.photos/id/23/200/300', # utensil
        'https://picsum.photos/id/24/200/300', # book
        'https://picsum.photos/id/31/200/300', # feet
        'https://picsum.photos/id/30/200/300', # cup
        'https://picsum.photos/id/64/200/300', # woman
        'https://picsum.photos/id/89/200/300', # grass
        'https://picsum.photos/id/106/200/300', # flower
    ]



    imgs = download_images(IMG_URLS)
    batch = prepare_batch(imgs)  # pick first image for inference

    # Load PyTorch model
    weights = ResNet50_Weights.DEFAULT
    torch_model = resnet50(weights=weights).to('cpu')
    torch_model.eval()


    torch_result = []
    t0 = time.perf_counter()
    for item in batch:
        label, score = inference(torch_model, item)
        torch_result.append((label, score))

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"PyTorch total elapsed: {elapsed:.4f}s")

    # -------------------------------------------

    ov_core = ov.Core()
    ov_model = ov_core.read_model('./ckpts/model.xml')

    ov_compiled = ov_core.compile_model(ov_model, device_name="CPU")

    ov_result = []
    t0 = time.perf_counter()
    for item in batch:
        label, score = inference(ov_compiled, item)
        ov_result.append((label, score))
    t1 = time.perf_counter()
    elapsed = t1 - t0

    print(f"OpenVINO total elapsed: {elapsed:.4f}s")



    # #----------------------------------------
    ov_core = ov.Core()
    q_ov_model = ov_core.read_model('./ckpts/quantized_model.xml')

    q_ov_compiled = ov_core.compile_model(q_ov_model, device_name="CPU")


    t0 = time.perf_counter()
    q_ov_result = []
    for item in batch:
        label, score = inference(q_ov_compiled, item)
        q_ov_result.append((label, score))
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"8bit-Quantized OpenVINO total elapsed: {elapsed:.4f}s")


    for (label_t, score_t), (label_ov, score_ov), (label_q, score_q) in zip(torch_result, ov_result, q_ov_result):
        print(f"| PyTorch: {label_t} - {score_t:.4f} | OpenVINO: {label_ov} - {score_ov:.4f} | Quantized OVI: {label_q} - {score_q:.4f}")

    
