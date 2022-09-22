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
    python3 -m pip install -r requirements.txt
}

install_deps() {
    # Basic python-related deps
    $APT update --yes
    $APT install python3-pip $APPEND
    $APT install python-is-python3 python3-autopep8 pylint $APPEND
    $APT install gcc python3-dev $APPEND
    $APT install make build-essential $APPEND

    # PAPI related deps
    $APT install papi-tools $APPEND
    python3 -m pip install python_papi

    # Hardware topology related deps
    $APT install hwloc libhwloc-dev numactl $APPEND

}

install_rust()
{
  if [ -f $HOME/.cargo/env ]; then
    source $HOME/.cargo/env
  fi

  # Make sure rust is up-to-date
  if [ ! -x "$(command -v rustup)" ] ; then
      curl https://sh.rustup.rs -sSf | sh -s -- -y
  fi

  source $HOME/.cargo/env
  rustup default nightly
  rustup component add rust-src
  rustup update
}

system_settings() {
  # Disable all the NUMA-related Linux policies
  $SUDO sh -c "echo 0 > /proc/sys/kernel/numa_balancing"
  $SUDO sh -c "echo 0 > /sys/kernel/mm/ksm/run"
  $SUDO sh -c "echo 0 > /sys/kernel/mm/ksm/merge_across_nodes"
  $SUDO sh -c "echo never > /sys/kernel/mm/transparent_hugepage/enabled"

  # Disable DVFS
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  $SUDO sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"

  # Disable Hyperthreading
  $SUDO sh -c "echo off > /sys/devices/system/cpu/smt/control"

  # Allow performance counters to be read
  $SUDO sh -c "echo 1 > /proc/sys/kernel/perf_event_paranoid"
}

# Check the number of arguments
USAGE="Usage: $0 [deps|rust|system|torch|vtune]"
if [ $# -ne 1 ]; then
  echo $USAGE
  exit 1
fi

if [ "$1" == "deps" ]; then
  install_deps
elif [ "$1" == "rust" ]; then
  install_rust
elif [ "$1" == "system" ]; then
  system_settings
elif [ "$1" == "torch" ]; then
  install_torch
elif [ "$1" == "vtune" ]; then
  install_vtune
else
  echo $USAGE
  exit 1
fi
