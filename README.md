# NUMA-aware Frameworks

## Setup
```bash
bash scripts/setup.sh deps
bash scripts/setup.sh torch
```

## Run
```bash
python run.py
```

## Build and Run docker image
```bash
docker build -t bench .
docker run -v `pwd`:/app -it bench:latest
```
