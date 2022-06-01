# Download base image ubuntu 20.04
# Build using `docker build -t bench .`
# Run using `docker run -it bench`

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

# Copy files from host to container
RUN mkdir /app
COPY . /app
WORKDIR /app

# Install dependencies
RUN /bin/bash scripts/setup.sh deps
RUN /bin/bash scripts/setup.sh torch

# Run the program
CMD [ "python3", "run.py"]
