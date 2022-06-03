# Download base image ubuntu 20.04
# Build using `docker build -t bench .`
# Run using ` docker run -v `pwd`:/app -it bench:latest`

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
COPY scripts/setup.sh setup.sh
RUN /bin/bash setup.sh deps
RUN /bin/bash setup.sh torch

# Set the working directory
WORKDIR /app

# Run the program
CMD [ "python3", "run.py" ]
