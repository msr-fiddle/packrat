# This benchmark is an extension to the benchmark explained at
# https://pytorch.org/hub/pytorch_vision_resnet/

import os
import time
import urllib
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms


def inference(model, data):
    with torch.no_grad():
        for _ in range(100):
            model(data)

        num_threads = torch.get_num_threads()
        omp_threads = os.environ.get("OMP_NUM_THREADS")
        assert num_threads == int(omp_threads)

        # TODO: Use benchmark APIs
        # https://pytorch.org/tutorials/recipes/recipes/benchmark.html
        start_time = time.time()
        for _ in range(100):
            _output = model(data)
        end_time = time.time()
        print('Threads {:d}, Time {:.2f} ms'.format(
            num_threads, (end_time - start_time) / 100 * 1000))


def get_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model


def get_test_data():
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
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def start():
    model = get_model()
    data = get_test_data()
    inference(model, data)


if __name__ == '__main__':
    start()
