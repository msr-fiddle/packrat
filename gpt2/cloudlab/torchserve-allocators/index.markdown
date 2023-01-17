---
layout: scalebench
title:  "GPT-2"
---

# Latency
## Latency Per Batch
<p float="left">
<img src="./gpt2-latency-1.png" alt="GPT2 Latency BS=1" width="100%"/>
<img src="./gpt2-latency-2.png" alt="GPT2 Latency BS=2" width="100%"/>
<img src="./gpt2-latency-4.png" alt="GPT2 Latency BS=4" width="100%"/>
<img src="./gpt2-latency-8.png" alt="GPT2 Latency BS=8" width="100%"/>
<img src="./gpt2-latency-16.png" alt="GPT2 Latency BS=16" width="100%"/>
<img src="./gpt2-latency-32.png" alt="GPT2 Latency BS=32" width="100%"/>
<img src="./gpt2-latency-64.png" alt="GPT2 Latency BS=64" width="100%"/>
<img src="./gpt2-latency-128.png" alt="GPT2 Latency BS=128" width="100%"/>
</p>

# Throughput
## Throughput Per Batch
<p float="left">
<img src="./gpt2-throughput-1.png" alt="GPT2 Throughput BS=1" width="100%"/>
<img src="./gpt2-throughput-2.png" alt="GPT2 Throughput BS=2" width="100%"/>
<img src="./gpt2-throughput-4.png" alt="GPT2 Throughput BS=4" width="100%"/>
<img src="./gpt2-throughput-8.png" alt="GPT2 Throughput BS=8" width="100%"/>
<img src="./gpt2-throughput-16.png" alt="GPT2 Throughput BS=16" width="100%"/>
<img src="./gpt2-throughput-32.png" alt="GPT2 Throughput BS=32" width="100%"/>
<img src="./gpt2-throughput-64.png" alt="GPT2 Throughput BS=64" width="100%"/>
<img src="./gpt2-throughput-128.png" alt="GPT2 Throughput BS=128" width="100%"/>
</p>

## Throughput Per Allocator
### (Allocator = default) 
<p float="left">
<img src="./gpt2-throughput-script-perBS.png" alt="GPT2 Throughput per BS" width="100%"/>
</p>

### (Allocator = tcmalloc) 
<p float="left">
<img src="./gpt2-throughput-script-perBS-tcmalloc.png" alt="GPT2 Throughput per BS" width="100%"/>
</p>