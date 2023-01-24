# Benchmarks

## PyTorch
```python
python run.py --help
```


### Single Instance
```python
python run.py --benchmark {resnet,inception,gpt2,bert} --allocator {default,tcmalloc}
```

### Multiple Instances

```python
python run.py --benchmark {resnet,inception,gpt2,bert} --allocator {default,tcmalloc} --instance_id 2
```

Multi-instance execution uses [Optimizer](./utils/optimizer.py) internally.
Initialize optimizer with appropriate parameters:

```python
optimizer = Optimizer(framework="pytorch", model=model, allocator="default",
                          optimization="script", profile_tag="large-batches")
```

where
- `framework` can be `pytorch` or `torchserve`
- `model` can be `resnet`, `inception`, `gpt2` or `bert`,
- `allocator` can be `default` or `tcmalloc`,
- `optimization` can be `script` or `none`,
- `profile_tag` is the benchmark profile tag on github.io.

## TorchServe
```python
cd torchserve
python scripts/benchmark-ab.py --help
```

### Single Instance
```python
python scripts/benchmark-ab.py --model {resnet,inception,gpt2,bert} --allocator {default,tcmalloc}
```

### Multiple Instances

```python
python scripts/benchmark-ab.py --model {resnet,inception,gpt2,bert} --allocator {default,tcmalloc} --multiinstance true
```

Also update optimizer parameters.
