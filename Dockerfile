# Download base image ubuntu 20.04
# Build using `docker build -t bench .`
# Run using ` docker run -v `pwd`:/app -it bench:latest`

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles
ENV PYTHONUNBUFFERED TRUE

RUN mkdir -p app
COPY . app/

# Setup for pytorch benchmarks
WORKDIR app/
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils
RUN /bin/bash scripts/setup.sh deps
RUN /bin/bash scripts/setup.sh torch
RUN /bin/bash scripts/setup.sh torchserve

ENV LD_LIBRARY_PATH=/opt/intel/ipp/lib

# Run the program
CMD [ "bash" ]
