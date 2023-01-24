import csv
from glob import glob
import json
import os
from pathlib import Path
from posixpath import expanduser
import re
import shutil
import sys
import tempfile
import time
from subprocess import PIPE, Popen
from enum import Enum
from urllib.parse import urlparse

import click
import click_config_file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import fcntl

default_ab_params = {
    "model": "resnet-18",
    "allocator": "tcmalloc",
    "multiinstance": False,
    "url": "https://torchserve.pytorch.org/mar_files/resnet-18.mar",
    "gpus": "",
    "exec_env": "local",
    "batch_size": 1,
    "batch_delay": 200,
    "workers": 1,
    "concurrency": 10,
    "requests": 100,
    "input": "dog.jpg",
    "content_type": "application/jpg",
    "image": "",
    "docker_runtime": "",
    "backend_profiling": False,
    "config_properties": "config.properties",
    "inference_model_url": "predictions/benchmark",
    "report_location": tempfile.gettempdir(),
    "tmp_dir": tempfile.gettempdir(),
}

execution_params = default_ab_params.copy()


def json_provider(file_path, cmd_name):
    with open(file_path) as config_data:
        return json.load(config_data)


class Models(Enum):
    resnet50 = 1
    inception = 2
    bert = 3
    gpt2 = 4
    mobilenet = 5
    squeezenet = 6


transformer_setting = {
    "model_name": "gpt2",
    "mode": "text_generation",
    "do_lower_case": True,
    "num_labels": 0,
    "save_mode": "pretrained",
    "max_length": 150,
    "captum_explanation": False,
    "FasterTransformer": False,
    "model_parallel": False,
    "embedding_name": "gpt2"
}


@click.command()
@click.argument("test_plan", default="custom")
@click.option(
    "--model",
    "-m",
    default=Models.resnet50.name,
    help=f"Input model name {[model.name for model in Models]}",
)
@click.option(
    "--allocator",
    "-a",
    default="tcmalloc",
    help="Pick the allocator from [default, tcmalloc]"
)
@click.option(
    "--multiinstance",
    "-m",
    default=False,
    help="Enable to run torchserve with multi-instances"
)
@click.option(
    "--url",
    "-u",
    default="https://torchserve.pytorch.org/mar_files/resnet-18.mar",
    help="Input model url",
)
@click.option(
    "--exec_env",
    "-e",
    type=click.Choice(["local", "docker"], case_sensitive=False),
    default="local",
    help="Execution environment",
)
@click.option(
    "--gpus",
    "-g",
    default="",
    help="Number of gpus to run docker container with.  Leave empty to run CPU based docker container",
)
@click.option(
    "--concurrency", "-c", default=10, help="Number of concurrent requests to run"
)
@click.option("--requests", "-r", default=100, help="Number of requests")
@click.option("--batch_size", "-bs", default=1, help="Batch size of model")
@click.option("--batch_delay", "-bd", default=200, help="Batch delay of model")
@click.option(
    "--input",
    "-i",
    default="dog.jpg",
    help="The input file path for model",
)
@click.option(
    "--content_type", "-ic", default="application/jpg", help="Input file content type"
)
@click.option("--workers", "-w", default=1, help="Number model workers")
@click.option(
    "--image", "-di", default="", help="Use custom docker image for benchmark"
)
@click.option(
    "--docker_runtime", "-dr", default="", help="Specify required docker runtime"
)
@click.option(
    "--backend_profiling",
    "-bp",
    default=False,
    help="Enable backend profiling using CProfile. Default False",
)
@click.option(
    "--config_properties",
    "-cp",
    default="config.properties",
    help="config.properties path, Default config.properties",
)
@click.option(
    "--inference_model_url",
    "-imu",
    default="predictions/benchmark",
    help="Inference function url - can be either for predictions or explanations. Default predictions/benchmark",
)
@click.option(
    "--report_location",
    "-rl",
    default=tempfile.gettempdir(),
    help=f"Target location of benchmark report. Default {tempfile.gettempdir()}",
)
@click.option(
    "--tmp_dir",
    "-td",
    default=tempfile.gettempdir(),
    help=f"Location for temporal files. Default {tempfile.gettempdir()}",
)
@click_config_file.configuration_option(
    provider=json_provider, implicit=False, help="Read configuration from a JSON file"
)
def start_benchmark(
    test_plan,
    model,
    allocator,
    multiinstance,
    url,
    gpus,
    exec_env,
    concurrency,
    requests,
    batch_size,
    batch_delay,
    input,
    workers,
    content_type,
    image,
    docker_runtime,
    backend_profiling,
    config_properties,
    inference_model_url,
    report_location,
    tmp_dir,
):
    input_params = {
        "model": model,
        "allocator": allocator,
        "multiinstance": multiinstance,
        "url": url,
        "gpus": gpus,
        "exec_env": exec_env,
        "batch_size": batch_size,
        "batch_delay": batch_delay,
        "workers": workers,
        "concurrency": concurrency,
        "requests": requests,
        "input": input,
        "content_type": content_type,
        "image": image,
        "docker_runtime": docker_runtime,
        "backend_profiling": backend_profiling,
        "config_properties": config_properties,
        "inference_model_url": inference_model_url,
        "report_location": report_location,
        "tmp_dir": tmp_dir,
    }

    # set ab params
    click.secho(f"Running test plan: {test_plan}", fg="green")
    update_exec_params(input_params)
    update_plan_params[test_plan]()


def benchmark():
    click.secho("Starting AB benchmark suite...", fg="green")
    click.secho("\n\nConfigured execution parameters are:", fg="green")
    click.secho(f"{execution_params}", fg="blue")

    # Setup execution env
    if execution_params["exec_env"] == "local":
        click.secho("\n\nPreparing local execution...", fg="green")
        local_torserve_start()
    else:
        click.secho("\n\nPreparing docker execution...", fg="green")
        docker_torchserve_start()

    check_torchserve_health()
    warm_up_lines = warm_up()
    run_benchmark()
    return generate_report(warm_up_lines=warm_up_lines)


def check_torchserve_health():
    attempts = 3
    retry = 0
    click.secho("*Testing system health...", fg="green")
    while retry < attempts:
        try:
            resp = requests.get(execution_params["inference_url"] + "/ping")
            if resp.status_code == 200:
                click.secho(resp.text)
                return True
        except Exception as e:
            retry += 1
            time.sleep(3)
    failure_exit(
        "Could not connect to Torchserve instance at "
        + execution_params["inference_url"]
    )


def warm_up():
    register_model()

    if is_workflow(execution_params["url"]):
        execution_params["inference_model_url"] = "wfpredict/{}".format(
            execution_params["model"])
    else:
        execution_params["inference_model_url"] = "predictions/{}".format(
            execution_params["model"])

    click.secho("\n\nExecuting warm-up ...", fg="green")

    ab_cmd = (
        f"ab -s 120 -c {execution_params['concurrency']}  -n {execution_params['requests']/10} -k -p "
        f"{execution_params['tmp_dir']}/benchmark/input -T  {execution_params['content_type']} "
        f"{execution_params['inference_url']}/{execution_params['inference_model_url']} > "
        f"{execution_params['result_file']}"
    )
    execute(ab_cmd, wait=True)

    warm_up_lines = sum(1 for _ in open(execution_params["metric_log"]))

    return warm_up_lines


def run_benchmark():
    if is_workflow(execution_params["url"]):
        execution_params["inference_model_url"] = "wfpredict/{}".format(
            execution_params["model"])
    else:
        execution_params["inference_model_url"] = "predictions/{}".format(
            execution_params["model"])

    click.secho("\n\nExecuting inference performance tests ...", fg="green")
    ab_cmd = (
        f"ab -s 120 -c {execution_params['concurrency']}  -n {execution_params['requests']} -k -p "
        f"{execution_params['tmp_dir']}/benchmark/input -T  {execution_params['content_type']} "
        f"{execution_params['inference_url']}/{execution_params['inference_model_url']} > "
        f"{execution_params['result_file']}"
    )
    execute(ab_cmd, wait=True)

    unregister_model()
    stop_torchserve()


def register_model():
    click.secho("*Registering model...", fg="green")
    if is_workflow(execution_params["url"]):
        url = execution_params["management_url"] + "/workflows"
        data = {
            "workflow_name": execution_params["model"],
            "url": execution_params["url"],
            "batch_delay": execution_params["batch_delay"],
            "batch_size": execution_params["batch_size"],
            "initial_workers": execution_params["workers"],
            "synchronous": "true",
        }
    else:
        url = execution_params["management_url"] + "/models"
        data = {
            "model_name": execution_params["model"],
            "url": execution_params["url"],
            "batch_delay": execution_params["batch_delay"],
            "batch_size": execution_params["batch_size"],
            "initial_workers": execution_params["workers"],
            "synchronous": "true",
        }
    resp = requests.post(url, params=data)
    if not resp.status_code == 200:
        failure_exit(f"Failed to register model.\n{resp.text}")
    click.secho(resp.text)


def unregister_model():
    click.secho("*Unregistering model ...", fg="green")
    if is_workflow(execution_params["url"]):
        resp = requests.delete(
            execution_params["management_url"] +
            "/workflows/" + execution_params["model"]
        )
    else:
        resp = requests.delete(
            execution_params["management_url"] + "/models/" + execution_params["model"])
    if not resp.status_code == 200:
        failure_exit(f"Failed to unregister model. \n {resp.text}")
    click.secho(resp.text)


def execute(command, wait=False, stdout=None, stderr=None, shell=True):
    print(command)
    cmd = Popen(
        command,
        shell=shell,
        close_fds=True,
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True,
    )
    if wait:
        cmd.wait()
    return cmd


def execute_return_stdout(cmd):
    proc = execute(cmd, stdout=PIPE)
    return proc.communicate()[0].strip()


def local_torserve_start():
    click.secho("*Terminating any existing Torchserve instance ...", fg="green")
    execute("torchserve --stop", wait=True)
    click.secho("*Setting up model store...", fg="green")
    prepare_local_dependency()
    click.secho("*Starting local Torchserve instance...", fg="green")

    execute(
        f"torchserve --start --model-store {execution_params['tmp_dir']}/model_store "
        f"--workflow-store {execution_params['tmp_dir']}/wf_store "
        f"--ts-config {execution_params['tmp_dir']}/benchmark/conf/{execution_params['config_properties_name']} "
        f"> {execution_params['tmp_dir']}/benchmark/logs/model_metrics.log"
    )

    time.sleep(3)


def docker_torchserve_start():
    prepare_docker_dependency()
    enable_gpu = ""
    if execution_params["image"]:
        docker_image = execution_params["image"]
        if execution_params["gpus"]:
            enable_gpu = f"--gpus {execution_params['gpus']}"
    else:
        if execution_params["gpus"]:
            docker_image = "pytorch/torchserve:latest-gpu"
            enable_gpu = f"--gpus {execution_params['gpus']}"
        else:
            docker_image = "pytorch/torchserve:latest"
        execute(f"docker pull {docker_image}", wait=True)

    backend_profiling = ""
    if execution_params["backend_profiling"]:
        backend_profiling = "-e TS_BENCHMARK=True"

    # delete existing ts container instance
    click.secho("*Removing existing ts container instance...", fg="green")
    execute("docker rm -f ts", wait=True)

    click.secho(
        f"*Starting docker container of image {docker_image} ...", fg="green")
    inference_port = urlparse(execution_params["inference_url"]).port
    management_port = urlparse(execution_params["management_url"]).port
    docker_run_cmd = (
        f"docker run {execution_params['docker_runtime']} {backend_profiling} --name ts --user root -p "
        f"{inference_port}:{inference_port} -p {management_port}:{management_port} "
        f"-v {execution_params['tmp_dir']}:/tmp {enable_gpu} -itd {docker_image} "
        f'"torchserve --start --model-store /home/model-server/model-store '
        f"\--workflow-store /home/model-server/wf-store "
        f"--ts-config /tmp/benchmark/conf/{execution_params['config_properties_name']} > "
        f'/tmp/benchmark/logs/model_metrics.log"'
    )
    execute(docker_run_cmd, wait=True)
    time.sleep(5)


def prepare_local_dependency():
    os.makedirs(
        os.path.join(execution_params["tmp_dir"], "model_store/"), exist_ok=True
    )
    shutil.rmtree(
        os.path.join(execution_params["tmp_dir"], "wf_store/"), ignore_errors=True
    )
    os.makedirs(os.path.join(
        execution_params["tmp_dir"], "wf_store/"), exist_ok=True)
    prepare_common_dependency()


def prepare_docker_dependency():
    prepare_common_dependency()


def prepare_common_dependency():
    input = execution_params["input"]
    shutil.rmtree(
        os.path.join(execution_params["tmp_dir"], "benchmark"), ignore_errors=True
    )
    shutil.rmtree(
        os.path.join(execution_params["report_location"], "benchmark"),
        ignore_errors=True,
    )
    os.makedirs(
        os.path.join(execution_params["tmp_dir"], "benchmark", "conf"), exist_ok=True
    )
    os.makedirs(
        os.path.join(execution_params["tmp_dir"], "benchmark", "logs"), exist_ok=True
    )
    os.makedirs(
        os.path.join(execution_params["report_location"], "benchmark"), exist_ok=True
    )

    shutil.copy(
        execution_params["config_properties"],
        os.path.join(execution_params["tmp_dir"], "benchmark", "conf"),
    )
    shutil.copyfile(
        input, os.path.join(execution_params["tmp_dir"], "benchmark", "input")
    )


def getAPIS():
    MANAGEMENT_API = "http://127.0.0.1:8081"
    INFERENCE_API = "http://127.0.0.1:8080"

    with open(execution_params["config_properties"], "r") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if "management_address" in line:
            MANAGEMENT_API = line.split("=")[1]
        if "inference_address" in line:
            INFERENCE_API = line.split("=")[1]

    execution_params["inference_url"] = INFERENCE_API
    execution_params["management_url"] = MANAGEMENT_API
    execution_params["config_properties_name"] = (
        execution_params["config_properties"].strip().split("/")[-1]
    )


def update_exec_params(input_param):
    for k, v in input_param.items():
        if default_ab_params[k] != input_param[k]:
            execution_params[k] = input_param[k]
    execution_params["result_file"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "result.txt"
    )
    execution_params["metric_log"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "logs", "model_metrics.log"
    )

    getAPIS()


def generate_report(warm_up_lines):
    click.secho("\n\nGenerating Reports...", fg="green")
    extract_metrics(warm_up_lines=warm_up_lines)
    output = generate_csv_output()
    generate_latency_output()
    generate_latency_graph()
    generate_profile_graph()
    click.secho("\nTest suite execution complete.", fg="green")
    return output


metrics = {
    "predict.txt": "PredictionTime",
    "handler_time.txt": "HandlerTime",
    "waiting_time.txt": "QueueTime",
    "worker_thread.txt": "WorkerThreadTime",
    "cpu_percentage.txt": "CPUUtilization",
    "memory_percentage.txt": "MemoryUtilization",
    "gpu_percentage.txt": "GPUUtilization",
    "gpu_memory_percentage.txt": "GPUMemoryUtilization",
    "gpu_memory_used.txt": "GPUMemoryUsed",
}


def extract_metrics(warm_up_lines):
    with open(execution_params["metric_log"]) as f:
        lines = f.readlines()

    click.secho(f"Dropping {warm_up_lines} warmup lines from log", fg="green")
    lines = lines[warm_up_lines:]

    for k, v in metrics.items():
        all_lines = []
        pattern = re.compile(v)
        for line in lines:
            if pattern.search(line):
                all_lines.append(line.split("|")[0].split(":")[3].strip())

        out_fname = os.path.join(
            *(execution_params["tmp_dir"], "benchmark", k))
        click.secho(
            f"\nWriting extracted {v} metrics to {out_fname} ", fg="green")
        with open(out_fname, "w") as outf:
            all_lines = map(lambda x: x + "\n", all_lines)
            outf.writelines(all_lines)


def generate_latency_output():
    batch_size = execution_params["concurrency"]
    with open(execution_params["metric_log"]) as f:
        lines = f.readlines()

    all_lines = []
    pattern = re.compile("predictions")
    for line in lines:
        if pattern.search(line):
            splitted = line.split(" ")
            latency = splitted[11].strip()
            all_lines.append(int(latency))

    # sum list elements in chunk of batch_size
    warm_up_lines = int(execution_params['requests']/10)
    inference_lines = all_lines[warm_up_lines:]
    print(len(inference_lines))
    final = []
    for i in range(0, len(inference_lines), batch_size):
        chunk = all_lines[i:i + batch_size]
        final.append(int(sum(chunk) / batch_size))
    out_fname = os.path.join(
        *(execution_params["tmp_dir"], "benchmark", "latency.txt"))
    #assert len(final) == 100
    with open(out_fname, "w") as outf:
        final = map(lambda x: str(x) + "\n", final)
        outf.writelines(final)


def generate_csv_output():
    click.secho("*Generating CSV output...", fg="green")
    batched_requests = execution_params["requests"] / \
        execution_params["batch_size"]
    line50 = int(batched_requests / 2)
    line90 = int(batched_requests * 9 / 10)
    line99 = int(batched_requests * 99 / 100)

    click.secho(
        f"Saving benchmark results to {execution_params['report_location']}")

    artifacts = {}
    with open(execution_params["result_file"]) as f:
        data = f.readlines()

    artifacts["Benchmark"] = "AB"
    artifacts["Batch size"] = execution_params["batch_size"]
    artifacts["Batch delay"] = execution_params["batch_delay"]
    artifacts["Workers"] = execution_params["workers"]
    artifacts["Model"] = "[.mar]({})".format(execution_params["url"])
    artifacts["Concurrency"] = execution_params["concurrency"]
    artifacts["Input"] = "[input]({})".format(execution_params["input"])
    artifacts["Requests"] = execution_params["requests"]
    artifacts["TS failed requests"] = extract_entity(
        data, "Failed requests:", -1)
    artifacts["TS throughput"] = extract_entity(
        data, "Requests per second:", -3)
    artifacts["TS latency P50"] = extract_entity(data, "50%", -1)
    artifacts["TS latency P90"] = extract_entity(data, "90%", -1)
    artifacts["TS latency P99"] = extract_entity(data, "99%", -1)
    artifacts["TS latency mean"] = extract_entity(
        data, "Time per request:.*mean\)", -3)
    if isinstance(artifacts["TS failed requests"], type(None)):
        artifacts["TS error rate"] = 0.0
    else:
        artifacts["TS error rate"] = (
            int(artifacts["TS failed requests"]) /
            execution_params["requests"] * 100
        )

    with open(
        os.path.join(execution_params["tmp_dir"], "benchmark", "predict.txt")
    ) as f:
        lines = f.readlines()
        lines.sort(key=float)
        artifacts["Model_p50"] = lines[line50].strip()
        artifacts["Model_p90"] = lines[line90].strip()
        artifacts["Model_p99"] = lines[line99].strip()

    for m in metrics:
        df = pd.read_csv(
            os.path.join(*(execution_params["tmp_dir"], "benchmark", m)),
            header=None,
            names=["data"],
        )
        if df.empty:
            artifacts[m.split(".txt")[0] + "_mean"] = 0.0
        else:
            artifacts[m.split(".txt")[0] +
                      "_mean"] = df["data"].values.mean().round(2)

    with open(
        os.path.join(
            execution_params["report_location"], "benchmark", "ab_report.csv"),
        "w",
    ) as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(artifacts.keys())
        csvwriter.writerow(artifacts.values())

    return artifacts


def extract_entity(data, pattern, index, delim=" "):
    pattern = re.compile(pattern)
    for line in data:
        if pattern.search(line):
            return line.split(delim)[index].strip()
    return None


def generate_latency_graph():
    click.secho("*Preparing graphs...", fg="green")
    df = pd.read_csv(
        os.path.join(execution_params["tmp_dir"], "benchmark", "predict.txt"),
        header=None,
        names=["latency"],
    )
    iteration = df.index
    latency = df.latency
    a4_dims = (11.7, 8.27)
    plt.figure(figsize=(a4_dims))
    plt.xlabel("Requests")
    plt.ylabel("Prediction time")
    plt.title("Prediction latency")
    plt.bar(iteration, latency)
    plt.savefig(
        f"{execution_params['report_location']}/benchmark/predict_latency.png")


def generate_profile_graph():
    click.secho("*Preparing Profile graphs...", fg="green")

    plot_data = {}
    for m in metrics:
        file_path = f"{execution_params['tmp_dir']}/benchmark/{m}"
        if is_file_empty(file_path):
            continue
        df = pd.read_csv(file_path, header=None)
        m = m.split(".txt")[0]
        plot_data[f"{m}_index"] = df.index
        plot_data[f"{m}_values"] = df.values

    if execution_params["requests"] > 100:
        sampling = int(execution_params["requests"] / 100)
    else:
        sampling = 1
    print(f"Working with sampling rate of {sampling}")

    a4_dims = (11.7, 8.27)
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2)
    plt.figure(figsize=a4_dims)
    fig1 = plt.subplot(grid[0, 0])
    fig2 = plt.subplot(grid[0, 1])
    fig3 = plt.subplot(grid[1, 0])
    fig4 = plt.subplot(grid[1, 1])
    fig5 = plt.subplot(grid[2, 0:])

    def plot_line(fig, data, color="blue", title=None):
        fig.set_title(title)
        fig.set_ylabel("Time (ms)")
        fig.set_xlabel("Percentage of queries")
        fig.grid()
        plot_points = np.arange(0, 100, 100 / len(data))
        x = plot_points[: len(data): sampling]
        y = data[::sampling]
        fig.plot(x, y, f"tab:{color}")

    # Queue Time
    plot_line(
        fig1, data=plot_data["waiting_time_values"], color="pink", title="Queue Time"
    )

    # handler Predict Time
    plot_line(
        fig2,
        data=plot_data["handler_time_values"],
        color="orange",
        title="Handler Time(pre & post processing + inference time)",
    )

    # Worker time
    plot_line(
        fig3,
        data=plot_data["worker_thread_values"],
        color="green",
        title="Worker Thread Time",
    )

    # Predict Time
    plot_line(
        fig4,
        data=plot_data["predict_values"],
        color="red",
        title="Prediction time(handler time+python worker overhead)",
    )

    # Plot in one graph
    plot_line(fig5, data=plot_data["waiting_time_values"], color="pink")
    plot_line(fig5, data=plot_data["handler_time_values"], color="orange")
    plot_line(fig5, data=plot_data["predict_values"], color="red")
    plot_line(
        fig5,
        data=plot_data["worker_thread_values"],
        color="green",
        title="Combined Graph",
    )
    fig5.grid()
    plt.savefig("api-profile1.png", bbox_inches="tight")


def stop_torchserve():
    if execution_params["exec_env"] == "local":
        click.secho("*Terminating Torchserve instance...", fg="green")
        execute("torchserve --stop", wait=True)
    else:
        click.secho("*Removing benchmark container 'ts'...", fg="green")
        execute("docker rm -f ts", wait=True)
    click.secho("Apache Bench Execution completed.", fg="green")


# Test plans (soak, vgg11_1000r_10c,  vgg11_10000r_100c,...)
def soak():
    execution_params["requests"] = 100000
    execution_params["concurrency"] = 10


def vgg11_1000r_10c():
    execution_params["url"] = "https://torchserve.pytorch.org/mar_files/vgg11.mar"
    execution_params["requests"] = 1000
    execution_params["concurrency"] = 10


def vgg11_10000r_100c():
    execution_params["url"] = "https://torchserve.pytorch.org/mar_files/vgg11.mar"
    execution_params["requests"] = 10000
    execution_params["concurrency"] = 100


def resnet152_batch():
    execution_params[
        "url"
    ] = "https://torchserve.pytorch.org/mar_files/resnet-152-batch.mar"
    execution_params["requests"] = 1000
    execution_params["concurrency"] = 10
    execution_params["batch_size"] = 4


def resnet152_batch_docker():
    execution_params[
        "url"
    ] = "https://torchserve.pytorch.org/mar_files/resnet-152-batch.mar"
    execution_params["requests"] = 1000
    execution_params["concurrency"] = 10
    execution_params["batch_size"] = 4
    execution_params["exec_env"] = "docker"


def bert_batch():
    execution_params[
        "url"
    ] = "https://torchserve.pytorch.org/mar_files/BERTSeqClassification.mar"
    execution_params["requests"] = 1000
    execution_params["concurrency"] = 10
    execution_params["batch_size"] = 4
    execution_params[
        "input"
    ] = "../examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text.txt"


def workflow_nmt():
    pass


def failure_exit(msg):
    click.secho(f"{msg}", fg="red")
    click.secho("Test suite terminated due to above failure", fg="red")
    sys.exit()


def is_workflow(model_url):
    return model_url.endswith(".war")


def is_file_empty(file_path):
    """Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def create_torchscripted_model(model):
    import torch
    from torchvision import models

    if model == Models.resnet50.name:
        eager_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        eager_model.eval()
    elif model == Models.mobilenet.name:
        eager_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT)
        eager_model.eval()
    elif model == Models.inception.name:
        eager_model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT)
        eager_model.eval()
    else:
        raise Exception(f"Unknown model: {model}")

    example_input = torch.rand(execution_params["batch_size"], 3, 224, 224)
    traced_script_module = torch.jit.trace(eager_model, example_input)
    traced_script_module.save(f"{model}.pt")


def create_mar(model):
    cmd = [
        "torch-model-archiver",
        f"--model-name {model}",
        "--version 1.0",
        f"--serialized-file {model}.pt",
        "--handler image_classifier",
        "--force",
    ]

    os.system(" ".join(cmd))
    os.remove(Path(os.getcwd()).joinpath(model + ".pt"))


def move_mar_file(model):
    mar_file = f"{model}.mar"
    mar_path = Path(os.getcwd()).joinpath(mar_file)
    model_store = Path(execution_params["tmp_dir"], "model_store/")
    if not os.path.exists(model_store):
        os.makedirs(model_store)

    shutil.move(mar_path, os.path.join(model_store, mar_file))


def download_input(url, filename: str):
    import urllib
    from PIL import Image

    try:
        urllib.request.urlretrieve(url, filename)
    except urllib.error.HTTPError:
        urllib.request.urlretrieve(url, filename)

    if filename.__contains__("jpg"):
        # TODO: Add to the vision_handler; if removed from here.
        img = Image.open(filename)
        img = img.resize((256, 256))
        img = img.crop((16, 16, 240, 240))
        img.save(filename)


def setup(model, image, filename):
    # Create torchscript based mar file for the given model
    create_torchscripted_model(model)
    create_mar(model)
    move_mar_file(model)

    # Download the sample input image
    download_input(image, filename)


def create_gpt2():
    import json
    shutil.rmtree(
        os.path.join("./", "Transformer_model/"), ignore_errors=True
    )

    model = Models.gpt2.name
    path = "setup_config.json"
    with open(path, 'w') as file:
        json_data = json.dumps(transformer_setting, indent=1)
        file.write(json_data)
        file.close()

        download = [
            "python",
            "./serve/examples/Huggingface_Transformers/Download_Transformer_models.py setup_config.json"
        ]
        os.system(" ".join(download))

    cmd = [
        "torch-model-archiver",
        f"--model-name {model}",
        "--version 1.0",
        "--serialized-file ./Transformer_model/pytorch_model.bin",
        "--handler ./serve/examples/Huggingface_Transformers/Transformer_handler_generalized.py",
        "--extra-files \"./Transformer_model/config.json,setup_config.json\""
    ]

    os.system(" ".join(cmd))
    move_mar_file(model)


def handle_model_specific_inputs():
    # Clean the model store
    shutil.rmtree(
        os.path.join(execution_params["tmp_dir"], "model_store/"), ignore_errors=True
    )
    model = execution_params["model"]
    if model in [Models.resnet50.name, Models.mobilenet.name, Models.inception.name]:
        input = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        filename = "dog.jpg"
        setup(model, input, filename)

        # Update the execution parameters
        execution_params["url"] = f"{model}.mar"
        execution_params["input"] = f"{filename}"
    elif model == Models.bert.name:
        execution_params["url"] = "https://torchserve.pytorch.org/mar_files/BERTSeqClassification_torchscript.mar"
        execution_params["input"] = "./serve/examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text.txt"
    elif model == Models.squeezenet.name:
        download_input(
            "https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/kitten.jpg", "kitten.jpg")
        execution_params["input"] = "kitten.jpg"
        execution_params["url"] = "https://torchserve.pytorch.org/mar_files/squeezenet1_1_scripted.mar"
    elif model == Models.gpt2.name:
        create_gpt2()
        execution_params["url"] = f"{model}.mar"
        execution_params["input"] = "./serve/examples/Huggingface_Transformers/Text_gen_artifacts/sample_text.txt"
        execution_params["content_type"] = "Application/text"
    else:
        raise Exception("Model is not handled yet!")


def update_config(cores):
    file_path = "config.properties"
    workers = execution_params["workers"]
    allocator = execution_params["allocator"]
    os.system(f'sed -i "$ d" {file_path}')

    use_multiinstance = ''
    use_allocator = ''
    if workers > 1:
        use_multiinstance = "--multi_instance"
    if allocator == "default":
        use_allocator = "--use_default_allocator"
    elif allocator == "tcmalloc":
        use_allocator = "--enable_tcmalloc"

        # Check if tcmalloc is installed
        library_paths = []
        library_paths += ["{}/.local/lib/".format(expanduser("~")), "/usr/local/lib/",
                          "/usr/local/lib64/", "/usr/lib/", "/usr/lib64/"]
        lib_type = "tcmalloc"
        lib_find = False
        for lib_path in library_paths:
            library_file = lib_path + "lib" + lib_type + ".so"
            matches = glob(library_file)
            if len(matches) > 0:
                lib_find = True
                break
        if not lib_find:
            click.secho("tcmalloc library not found...", fg="red")
            sys.exit(1)

    os.system(
        f"echo cpu_launcher_args={use_multiinstance} {use_allocator} --ninstances {workers} --ncore_per_instance {cores} --node_id 0 >> {file_path}")


def run_singe_instance(workers: int, cores: int, batch_size: int):
    print(workers, cores, batch_size)
    model = execution_params["model"]
    allocator = execution_params["allocator"]
    output_file = f"{model}_results.csv"
    exists: bool = os.path.isfile(output_file)
    file = open(output_file, "a+")
    writer = csv.writer(file, delimiter=",")
    if not exists:
        writer.writerow(["model", "instances", "cores", "batch_size", "allocator",
                         "throughput", "mean_ts_latency", "mean_predict_latency"])

    execution_params["batch_size"] = batch_size
    execution_params["workers"] = workers
    execution_params["batch_delay"] = 100 * \
        execution_params["batch_size"]
    execution_params["concurrency"] = execution_params["workers"] * \
        execution_params["batch_size"]
    execution_params["requests"] = execution_params["concurrency"] * 100

    update_config(cores)
    handle_model_specific_inputs()
    output = benchmark()
    fcntl.flock(file, fcntl.LOCK_EX)
    writer.writerow([model, workers, cores, output["Batch size"], allocator,
                     output["TS throughput"], output["TS latency mean"], output["predict_mean"]])
    file.flush()
    fcntl.flock(file, fcntl.LOCK_UN)
    file.close()


# For now manually run optimizer and store the results here
solution = {
    "resnet50": {
        "default": {8: 8, 16: 16, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16},
        "tcmalloc": {8: 8, 16: 16, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16}
    },
    "inception": {
        "default": {8: 8, 16: 8, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16},
        "tcmalloc": {8: 8, 16: 16, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16}
    },
    "bert": {
        "default": {8: 4, 16: 4, 32: 4, 64: 4, 128: 16, 256: 16, 512: 16, 1024: 16},
        "tcmalloc": {8: 4, 16: 4, 32: 4, 64: 8, 128: 16, 256: 16, 512: 16, 1024: 16}
    },
    "gpt2": {
        "default": {8: 4, 16: 8, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16},
        "tcmalloc": {8: 8, 16: 4, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16},
    },
    "mobilenet": {
        "default": {8: 8, 16: 16, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16},
        "tcmalloc": {8: 8, 16: 16, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16},
    },
    "squeezenet": {
        "default": {8: 8, 16: 16, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16},
        "tcmalloc": {8: 8, 16: 16, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16, 1024: 16},
    }
}


def custom():
    core_count = 16
    multiinstance = execution_params["multiinstance"]
    if not multiinstance:
        workers = 1
        for b in range(0, 8):
            for cores in range(1, core_count + 1):
                batch_size = 2 ** b
                run_singe_instance(workers, cores, batch_size)
    else:
        for b in range(3, 11):
            batch_size = 2 ** b
            # Fat instance
            run_singe_instance(1, core_count, batch_size)

            # Multi-instance
            workers = solution[execution_params["model"]
                               ][execution_params["allocator"]][batch_size]
            thin_cores = int(core_count / workers)
            thin_batch = int(batch_size / workers)
            run_singe_instance(workers, thin_cores, thin_batch)
            shutil.rmtree(
                os.path.join("./", "logs"), ignore_errors=True
            )


update_plan_params = {
    "soak": soak,
    "vgg11_1000r_10c": vgg11_1000r_10c,
    "vgg11_10000r_100c": vgg11_10000r_100c,
    "resnet152_batch": resnet152_batch,
    "resnet152_batch_docker": resnet152_batch_docker,
    "bert_batch": bert_batch,
    "workflow_nmt": workflow_nmt,
    "custom": custom,
}

if __name__ == "__main__":
    start_benchmark()
