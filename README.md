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

To run VTune inside docker image

```bash
docker run --pid=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE -v `pwd`:/app -it bench:latest
```

Reference: 
- [Run VTune Profiler in a Container](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/launch/containerization-support/run-from-container.html)
- [Install VTune with the Packet Managers](https://www.intel.com/content/www/us/en/develop/documentation/vtune-install-guide/top/linux/package-managers.html)
