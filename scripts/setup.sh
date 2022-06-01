#!/bin/bash

set -e

install_torch() {
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
}

install_deps() {
    sudo apt update --yes
    sudo apt install python3-pip --yes
    sudo apt install python-is-python3 python3-autopep8 pylint --yes
    sudo apt install numactl --yes
}

# Check the number of arguments
if [ $# -ne 1 ]; then
  echo "Usage: $0 <deps|torch>"
  exit 1
fi

if [ "$1" == "deps" ]; then
  install_deps
elif [ "$1" == "torch" ]; then
  install_torch
else
  echo "Unknown argument: $1"
  exit 1
fi