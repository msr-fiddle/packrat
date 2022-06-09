#!/bin/bash

set -e

if [ "$EUID" -ne 0 ]; then
    echo "not running as root, using sudo"
    APT="sudo apt-get"
    SUDO="sudo"
else
    echo "running as root."
    APT="apt-get"
    SUDO=""
fi

APPEND="--yes --no-install-recommends"

install_vtune() {
    $APT install wget gnupg $APPEND
    $APT install kmod software-properties-common pkg-config $APPEND
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    $SUDO apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB 2>/dev/null
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB || true
    $SUDO add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"

    $APT update --yes
    $APT install linux-headers-generic $APPEND
    $APT install intel-oneapi-vtune $APPEND
}

install_torch() {
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install psutil
    pip3 install pandas plotnine
}

install_deps() {
    $APT update --yes
    $APT install python3-pip $APPEND
    $APT install python-is-python3 python3-autopep8 pylint $APPEND
    $APT install numactl $APPEND
    $APT install gcc python3-dev $APPEND
    $APT install make build-essential $APPEND
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
elif [ "$1" == "vtune" ]; then
  install_vtune
else
  echo "Usage: $0 <deps|torch|vtune>"
  exit 1
fi