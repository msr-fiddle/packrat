#!/bin/bash

set -e

if [ "$EUID" -ne 0 ]; then
    echo "not running as root, using sudo"
    APT="sudo apt-get"
else
    echo "running as root."
    APT="apt-get"
fi

install_torch() {
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install psutil
}

install_deps() {
    APPEND="--yes --no-install-recommends"
    $APT update --yes
    $APT install python3-pip $APPEND
    $APT install python-is-python3 python3-autopep8 pylint $APPEND
    $APT install numactl $APPEND
    $APT install gcc python3-dev $APPEND
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