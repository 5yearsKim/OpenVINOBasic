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


model = ov.Core().read_model("./ckpts/model.xml")

quantized_model = nncf.quantize(model, calibration_dataset)

ov.save_model(quantized_model, "./ckpts/quantized_model.xml")
