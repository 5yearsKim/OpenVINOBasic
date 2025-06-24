import openvino as ov
import nncf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


calib_transform = transforms.Compose([
    transforms.Resize(224),    
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
all_ds = datasets.CIFAR10(root='./data', train=False, download=True,
                            transform=calib_transform)
calib_ds = Subset(all_ds, list(range(300)))

calibration_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)


def transform_fn(data_item):
    images, _ = data_item
    return images.numpy()

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)


# convert.py 에서 변환한 모델을 읽어 nncf 로 양자화 시켜보자.
# 양자화된 모델을 ./ckpts/quantized_model.xml 에 저장해보자.

# ---start---
# Fill your code
# ---end---
