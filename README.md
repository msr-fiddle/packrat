[![Build and test packrat modules](https://github.com/msr-fiddle/naf/actions/workflows/python.yml/badge.svg)](https://github.com/msr-fiddle/naf/actions/workflows/python.yml)

# Packrat
## Setup
```bash
bash scripts/setup.sh deps
bash scripts/setup.sh rust
bash scripts/setup.sh system
bash scripts/setup.sh torch
```

## Run
```bash
python run.py --help
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

## Various configurable options

### Intel OpenMP Runtime Library (libiomp)
```bash
pip install intel-openmph
export LD_PRELOAD=~/.local/lib/libiomp5.so
```

### Tcmalloc
```bash
sudo apt install google-perftools
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
```

### Jemalloc
```bash
sudo apt install libjemalloc-dev
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

### Vectorization
```bash
ATEN_CPU_CAPABILITY=default/avx/avx2/avx512
```
