---
layout: scalebench
title:  "Bert"
---

# Latency
## Latency Per Batch
<p float="left">
<img src="./bert-latency-1.png" alt="Bert Latency BS=1" width="100%"/>
<img src="./bert-latency-2.png" alt="Bert Latency BS=2" width="100%"/>
<img src="./bert-latency-4.png" alt="Bert Latency BS=4" width="100%"/>
<img src="./bert-latency-8.png" alt="Bert Latency BS=8" width="100%"/>
<img src="./bert-latency-16.png" alt="Bert Latency BS=16" width="100%"/>
<img src="./bert-latency-32.png" alt="Bert Latency BS=32" width="100%"/>
<img src="./bert-latency-64.png" alt="Bert Latency BS=64" width="100%"/>
<img src="./bert-latency-128.png" alt="Bert Latency BS=128" width="100%"/>
</p>

# Throughput
## Throughput Per Batch
<p float="left">
<img src="./bert-throughput-1.png" alt="Bert Throughput BS=1" width="100%"/>
<img src="./bert-throughput-2.png" alt="Bert Throughput BS=2" width="100%"/>
<img src="./bert-throughput-4.png" alt="Bert Throughput BS=4" width="100%"/>
<img src="./bert-throughput-8.png" alt="Bert Throughput BS=8" width="100%"/>
<img src="./bert-throughput-16.png" alt="Bert Throughput BS=16" width="100%"/>
<img src="./bert-throughput-32.png" alt="Bert Throughput BS=32" width="100%"/>
<img src="./bert-throughput-64.png" alt="Bert Throughput BS=64" width="100%"/>
<img src="./bert-throughput-128.png" alt="Bert Throughput BS=128" width="100%"/>
</p>

## Throughput Per Allocator
### (Allocator = default) 
<p float="left">
<img src="./bert-throughput-script-default-perBS.png" alt="Bert Throughput per BS" width="100%"/>
</p>

### (Allocator = tcmalloc) 
<p float="left">
<img src="./bert-throughput-script-tcmalloc-perBS.png" alt="Bert Throughput per BS" width="100%"/>
</p>