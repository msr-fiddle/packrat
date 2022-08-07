from math import log
from PIL import Image

import torch
import torchvision.models as models
from torchvision import transforms
import urllib


def get_test_data(batch_size: int) -> torch.Tensor:
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.request.urlretrieve(url, filename)
    except urllib.error.HTTPError:
        urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = []
    for i in range(0, batch_size, 1):
        input_batch.append(input_tensor)

    data = torch.stack(input_batch)
    return data


def get_model_size():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.eval()

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2

    print(f"Param size {param_size / 1024} KB")
    print(f"Buffer size {buffer_size / 1024} KB")
    print(f"Total size: {size_all_mb:.3f} MB\n")


if __name__ == "__main__":
    get_model_size()

    for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        tensor = get_test_data(int(batch))
        size = tensor.nelement() * tensor.element_size()
        print(f"BS {batch}, Size {size / 1024} KB")
