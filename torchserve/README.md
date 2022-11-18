# Notes

## Contents
- [Notes](#notes)
  - [Contents](#contents)
  - [TorchServe Internals](#torchserve-internals)
  - [Run TorchServe](#run-torchserve)
  - [Benchmarking](#benchmarking)
  - [Run TorchServe with the launcher](#run-torchserve-with-the-launcher)

## TorchServe Internals
Read about TorchServe Internals [here](https://github.com/ankit-iitb/serve/blob/release_0.6.1/docs/internals.md).

At a high-level TorchServe uses the following components:
- Frontend Server: This is the entry point for all inference requests. It uses Netty based NIO HTTP server to serve REST API requests. It also uses Netty based NIO WebSocket server to serve real time inference requests.

- Backend Workers: These are the worker processes that handle the actual inference requests. They are managed by the Frontend Server. The number of workers can be scaled up or down based on the load.

## Run TorchServe

**Start TorchServe**

```bash
torchserve --start --model-store model_store
```

**Create a model archive**

```bash
torch-model-archiver --model-name my_model --version 1.0 --model-file model.py --serialized-file model.pth --handler handler.py --extra-files index_to_name.json
```
- `model-file`: The file that contains the model definition. This file is required.
- `handler`: The file that contains the inference logic. This file is required.
  - The default handlers can be found [here](https://github.com/ankit-iitb/serve/tree/release_0.6.1/ts/torch_handler).
  - The default handler functionalities are described [here](https://github.com/ankit-iitb/serve/blob/release_0.6.1/docs/default_handlers.md).

Check all the options for model-archiver using `torch-model-archiver --help`.

**Register a model**

```bash
curl -X POST "http://localhost:8081/models?model_name=my_model&url=my_model.mar&initial_workers=1&synchronous=true"
```

**Run inference**

```bash
curl -X POST http://localhost:8080/predictions/my_model -T kitten.jpg
```

**Scale workers**

```bash
curl -X PUT "http://localhost:8081/models/my_model?min_worker=1&max_worker=5&synchronous=true"
```

**Unregister a model**

```bash
curl -X DELETE "http://localhost:8081/models/my_model"
```

**Stop TorchServe**

```bash
torchserve --stop
```

[**Batch Inference with TorchServe**](https://github.com/ankit-iitb/serve/blob/release_0.6.1/docs/batch_inference_with_ts.md)

## Benchmarking
TODO

## Run TorchServe with the launcher
TODO