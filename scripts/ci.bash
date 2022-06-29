#!/bin/bash

set -ex
eval `keychain --agents ssh --eval id_rsa_ankit`

# Run benchmarks
# python3 run.py

# Copy data and graphs
CI_MACHINE_TYPE=skylake2x
export GIT_REV_CURRENT=`git rev-parse --short HEAD`
export CSV_LINE="`date +%Y-%m-%d`",${GIT_REV_CURRENT},"","index.html"

# Check that we can checkout gh-pages early:
rm -rf gh-pages
git clone --depth 1 git@github.com:msr-fiddle/numa-aware-frameworks.git -b gh-pages gh-pages
pip3 install -r gh-pages/requirements.txt

# plot the data
python3 scripts/plot.py latency resnet_latency.csv
python3 scripts/plot.py throughput resnet_throughput.csv

# Copy the data and graphs to the gh-pages repo
echo $CSV_LINE >> gh-pages/_data/${CI_MACHINE_TYPE}.csv
RESNET_DEPLOY="gh-pages/resnet/${CI_MACHINE_TYPE}/${GIT_REV_CURRENT}"
rm -rf ${RESNET_DEPLOY}
mkdir -p ${RESNET_DEPLOY}
mv *.csv ${RESNET_DEPLOY}
mv *.png ${RESNET_DEPLOY}

#gzip ${RESNET_DEPLOY}/resnet_benchmarks.csv
cp gh-pages/resnet/index.markdown ${RESNET_DEPLOY}

# Push to gh-pages
cd gh-pages
git add . -f
git commit -a -m "Added benchmark results for $GIT_REV_CURRENT."
git push
cd ..

rm -rf gh-pages
