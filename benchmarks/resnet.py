# This benchmark is an extension to the benchmark explained at
# https://pytorch.org/hub/pytorch_vision_resnet/

import csv
import math
import os
import subprocess
import time
import urllib
from PIL import Image
import psutil
import torch
import torchvision.models as models
from torchvision import transforms


def inference(model, data):
    sockets = int(subprocess.check_output(
        'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
    cores_per_socket = psutil.cpu_count(logical=False) / sockets

    num_threads = int(os.environ.get("OMP_NUM_THREADS"))
    if num_threads <= cores_per_socket:
        torch.set_num_threads(num_threads)
    else:
        torch.set_num_threads(2**int(math.log(cores_per_socket, 2)))
    torch.set_num_interop_threads(num_threads)

    # open the csv file for writing
    filename = "resnet_benchmark.csv"
    exist = os.path.exists(filename)
    writer = csv.writer(open(filename, "a+"), delimiter=",")
    if not exist:
        writer.writerow(["threads", "latency"])

    with torch.no_grad():
        for _ in range(100):
            model(data)

    # TODO: Use benchmark APIs
    # https://pytorch.org/tutorials/recipes/recipes/benchmark.html
    start_time = time.time()
    for _ in range(100):
        _output = model(data)
    end_time = time.time()
    average_latency = ((end_time - start_time) * 1000) / 100
    print('Threads {:d}, Time {:.2f} ms'.format(
        num_threads, average_latency))
    writer.writerow([num_threads, average_latency])


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
