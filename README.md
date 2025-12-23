# LLM Load Testing Framework (Data Collection for LLM Performance and Quality Degradation Analysis)

A comprehensive load testing framework designed to evaluate Large Language Model (LLM) performance and reasoning integrity under varying operational stress. This project provides the tooling and methodology to benchmark inference latency and output accuracy across diverse model architectures.

This framework quantifies how high-concurrency environments and massive context loads impact the reliability of LLMs. By analyzing the relationship between system resource saturation (VRAM/GPU utilization) and model quality, the project offers insights into the trade-offs between serving efficiency and contextual fidelity.

Two types of models were used for this research
1. **Dense Transformer Models (Primary):** Llama-3.1-70B and Qwen-1.5-14B models are used to cover LLMs of small and large sizes.
2. **Sparse Mixture-of-Experts (MOE):** Mistral (Mixtral-8x7B MoE model with ~13B active
parameters) is utilized as an architectural extension to compare how sparse activation routing handles high-concurrency stress differently than traditional dense models.
---

## Prerequisites

### 1. Hugging Face API Token (Required)

You must obtain a Hugging Face API token with WRITE permissions to access the models.

Add this to your deployment setup.


##  Project Structure ##

```
LOAD_TESTER_FILES/
│
├── LLAMA/                          # LLaMA Model Tests
│   ├── Baseline_test.py            # Baseline accuracy test (3 loops, Zero Context)
│   ├── Thread_test.py              # Threading load test (Not performed due to the size of the model)
│   └── server_start.sh             # vLLM server startup script
│
├── QWEN/                           # Qwen Model Tests
│   ├── Baseline_test.py            # Baseline accuracy test (3 loops, Zero Context)
│   ├── Thread_test.py              # Threading load test (4096 context, max 4 concurrent threads)
│   └── server_start.sh             # vLLM server startup script
│
├── MISTRAL/                        # Mistral Model Tests
│   ├── Baseline_test.py            # Baseline accuracy test (3 loops, Zero Context)
│   ├── Thread_test.py              # Threading load test (4096 context, max 4 concurrent threads)
│   └── server_start.sh             # vLLM server startup script
│
└── README.md                       
```
## Execution Environment ##

The tests were run in the following server configuration. 
The AWS **p4d.24xlarge instance** is a high-performance computing instance designed for training and deploying large-scale machine learning models.

### The server configuration

- **GPUs:** 8 NVIDIA A100 Tensor Core GPUs.
- **GPU Memory (VRAM):** 40 GB per GPU (Total 320 GB VRAM across the instance).
- **System Memory (RAM):** 1.1 TB (1,152 GB).
- **vCPUs:** 96 vCPUs (Intel Xeon Scalable processors).
- **Network Bandwidth:** 400 Gbps.
