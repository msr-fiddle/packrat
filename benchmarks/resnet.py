"""
This benchmark is an extension to the benchmark explained at
https://pytorch.org/hub/pytorch_vision_resnet/
"""

import csv
import os
import sys
import time
import urllib
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms


def run_inference(model, data):
    return model(data)


def inference_manual(model: models.RegNet, data: torch.Tensor, batch_size: int):
    num_threads = int(os.environ.get("OMP_NUM_THREADS"))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    assert torch.get_num_threads, num_threads
    assert torch.get_num_interop_threads, num_threads

    # print(torch.__config__.parallel_info())

    # open the csv file for writing
    filename = "resnet_benchmark.csv"
    exist = os.path.exists(filename)
    writer = csv.writer(open(filename, "a+"), delimiter=",")
    if not exist:
        writer.writerow(["threads", "batch_size", "latency"])

    with torch.no_grad():
        for _ in range(100):
            run_inference(model, data)

    start_time = time.time()
    for _ in range(100):
        _output = run_inference(model, data)
    end_time = time.time()
    average_latency = ((end_time - start_time) * 1000) / 100
    print('Threads {:d}, Batch {}, Time {:.2f} ms'.format(
        num_threads, batch_size, average_latency))
    writer.writerow([num_threads, batch_size, average_latency])


def inference_benchmark(model, data):
    import torch.utils.benchmark as benchmark
    num_threads = int(os.environ.get("OMP_NUM_THREADS"))
    timer = benchmark.Timer(stmt="run_inference(model, data)",
                            setup="from __main__ import run_inference",
                            globals={
                                "model": model,
                                "data": data,
                            },
                            num_threads=num_threads,
                            label="Latency Measurement",
                            sub_label="torch.utils.benchmark.").blocked_autorange(min_run_time=5)

    print(
        f"Thread {num_threads}, Mean: {timer.mean * 1e3:6.2f} ms, Median: {timer.median * 1e3:6.2f} ms")


def get_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model


def get_test_data(batch_size: int):
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
    return data.to_mkldnn()


def start(r_type, batch_size: int):
    model = get_model()
    data = get_test_data(batch_size)

    if r_type == "default":
        inference_benchmark(model, data)
    elif r_type == "manual":
        inference_manual(model, data, batch_size)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 resnet.py default/manual batch_size")
        sys.exit(1)

    start(sys.argv[1], int(sys.argv[2]))
