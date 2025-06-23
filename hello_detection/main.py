# main.py 함수를 작성해서 original.py 함수와 같은 기능을 프로젝트화 해보자.

# ---start---

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from hello_detector import HelloDetector
from notebook_utils import download_file

base_model_dir = Path("./model").expanduser()

model_name = "horizontal-text-detection-0001"
model_xml_name = f"{model_name}.xml"
model_bin_name = f"{model_name}.bin"

model_xml_path = base_model_dir / model_xml_name


detector = HelloDetector(model_path=model_xml_path, device="CPU")


image_filename = Path("data/intel_rnb.jpg")

if not image_filename.exists():
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/intel_rnb.jpg",
        directory="data",
    )

# Text detection models expect an image in BGR format.
image = cv2.imread(str(image_filename))

boxes, resized_image = detector.detect(image)


img = detector.convert_result_to_image(image, resized_image, boxes, conf_labels=False)


plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis("off")

plt.show()

# ---end---