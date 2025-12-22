# LLM Load Testing Framework

A comprehensive load testing framework for evaluating Large Language Model (LLM) performance under various conditions. This framework tests baseline accuracy and threading capabilities for multiple model families including LLaMA, Mistral, and Qwen.

---

##Prerequisites

### 1. Hugging Face API Token (Required)

You must obtain a Hugging Face API token with WRITE permissions to access LLaMA models.

Add this to your deployment setup.


##  Project Structure

```
LOAD_TESTER_FILES/
│
├── LLAMA/                          # LLaMA Model Tests
│   ├── Baseline_test.py            # Baseline accuracy test (3 loops)
│   ├── Thread_test.py              # Threading load test (4096 context)
│   └── server_start.sh             # vLLM server startup script
│
├── MISTRAL/                        # Mistral Model Tests
│   ├── Baseline_test.py            # Baseline accuracy test (3 loops)
│   ├── Thread_test.py              # Threading load test (4096 context)
│   └── server_start.sh             # vLLM server startup script
│
├── QWEN/                           # Qwen Model Tests
│   ├── Baseline_test.py            # Baseline accuracy test (3 loops)
│   ├── Thread_test.py              # Threading load test (4096 context)
│   └── server_start.sh             # vLLM server startup script
│
└── README.md                       
