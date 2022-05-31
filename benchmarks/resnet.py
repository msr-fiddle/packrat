# This benchmark is an extension to the benchmark explained at
# https://pytorch.org/hub/pytorch_vision_resnet/

import torch
import torchvision.models as models
from torchvision import transforms

import time
import urllib
from PIL import Image


def inference(model, data):
    with torch.no_grad():
        for i in range(100):
            model(data)

        num_threads = torch.get_num_threads()

        # TODO: Use benchmark APIs
        # https://pytorch.org/tutorials/recipes/recipes/benchmark.html
        start = time.time()
        for i in range(100):
            output = model(data)
        end = time.time()
        print('Threads {:d}, Time {:.2f} ms'.format(
            num_threads, (end - start) / 100 * 1000))


def get_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model


def get_test_data():
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
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


def main():
    model = get_model()
    data = get_test_data()
    inference(model, data)


if __name__ == '__main__':
    main()