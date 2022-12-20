---
layout: scalebench
title:  "ResNet50"
---

# Latency
## Latency Per Batch
<p float="left">
<img src="./resnet50-latency-1.png" alt="ResNet Latency BS=1" width="100%"/>
<img src="./resnet50-latency-2.png" alt="ResNet Latency BS=2" width="100%"/>
<img src="./resnet50-latency-4.png" alt="ResNet Latency BS=4" width="100%"/>
<img src="./resnet50-latency-8.png" alt="ResNet Latency BS=8" width="100%"/>
<img src="./resnet50-latency-16.png" alt="ResNet Latency BS=16" width="100%"/>
<img src="./resnet50-latency-32.png" alt="ResNet Latency BS=32" width="100%"/>
<img src="./resnet50-latency-64.png" alt="ResNet Latency BS=64" width="100%"/>
<img src="./resnet50-latency-128.png" alt="ResNet Latency BS=128" width="100%"/>
</p>

# Throughput
## Throughput Per Batch
<p float="left">
<img src="./resnet50-throughput-1.png" alt="ResNet Throughput BS=1" width="100%"/>
<img src="./resnet50-throughput-2.png" alt="ResNet Throughput BS=2" width="100%"/>
<img src="./resnet50-throughput-4.png" alt="ResNet Throughput BS=4" width="100%"/>
<img src="./resnet50-throughput-8.png" alt="ResNet Throughput BS=8" width="100%"/>
<img src="./resnet50-throughput-16.png" alt="ResNet Throughput BS=16" width="100%"/>
<img src="./resnet50-throughput-32.png" alt="ResNet Throughput BS=32" width="100%"/>
<img src="./resnet50-throughput-64.png" alt="ResNet Throughput BS=64" width="100%"/>
<img src="./resnet50-throughput-128.png" alt="ResNet Throughput BS=128" width="100%"/>
</p>

## Throughput Per Allocator
### (Allocator = default) 
<p float="left">
<img src="./resnet50-throughput-script-perBS.png" alt="ResNet Throughput per BS" width="100%"/>
</p>

### (Allocator = tcmalloc) 
<p float="left">
<img src="./resnet50-throughput-script-perBS-tcmalloc.png" alt="ResNet Throughput per BS" width="100%"/>
</p>