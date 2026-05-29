# Instruction Fine-tuning: TinyLlama on Alpaca Dataset

Fine-tune TinyLlama-1.1B on the Alpaca instruction dataset using Unsloth for 5-8x faster training with minimal memory footprint. Perfect for learning instruction-following on resource-constrained devices.

---

## 🎯 Project Overview

This project demonstrates how to efficiently fine-tune **TinyLlama-1.1B** on instruction-following tasks using the Alpaca dataset. The model learns to follow complex instructions and generate appropriate responses.

### Key Technologies

- **[Unsloth](https://github.com/unslothai/unsloth)** - 5-8x faster fine-tuning with 80% less memory
- **[LoRA](https://arxiv.org/abs/2106.09685)** - Parameter-Efficient Fine-Tuning (PEFT)
- **[SFT](https://huggingface.co/docs/trl/sft_trainer)** - Supervised Fine-Tuning via TRL
- **[HuggingFace Hub](https://huggingface.co/)** - Model sharing and deployment
- **[4-bit QLoRA](https://arxiv.org/abs/2305.14314)** - Quantized LoRA for memory efficiency

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Base Model** | unsloth/tinyllama-bnb-4bit |
| **Model Size** | 1.1B parameters |
| **Dataset** | Alpaca (10,000 samples) |
| **Training Time** | ~15-25 minutes (single T4 GPU) |
| **Trainable Parameters** | 16.4M (1.49% of base) |
| **Batch Size** | 4 (Effective: 16 with gradient accumulation) |
| **Learning Rate** | 5e-4 |
| **Max Sequence Length** | 512 tokens |
| **LoRA Rank / Alpha** | 16 / 16 |
| **LoRA Dropout** | 0.05 |
| **Epochs** | 2 |
| **GPU Memory Used** | ~6-8 GB VRAM |

---

## 🚀 Quick Start

### Prerequisites

```bash
git clone https://github.com/basantapokharel/tinyllama-alpaca-instruction-finetuning.git
cd tinyllama-alpaca-instruction-finetuning
pip install -r requirements.txt
```

### Option 1: Google Colab (Recommended ⭐)

1. Open notebook in [Google Colab](https://colab.research.google.com/)
2. Upload `instruction_finetuning.ipynb`
3. Click `Runtime` → `Change runtime type` → Select **GPU (T4)**
4. Set your HF token in Colab Secrets (optional, for auto-upload)
5. Run all cells sequentially
6. Model automatically pushes to HuggingFace Hub

### Option 2: Local Machine

```python
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
import torch

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Basanta55/tinyllama-alpaca-qlora",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Switch to inference mode
FastLanguageModel.for_inference(model)

# Define prompt template
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

# Generate response
def generate(instruction: str, inp: str = "", max_tokens: int = 256):
    if inp:
        prompt = PROMPT_TEMPLATE.format(instruction=instruction, input=inp, response="")
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
response = generate("Explain machine learning in simple terms")
print(response)
```


---

## 🔧 Configuration

Modify these settings in the notebook's **Configuration Cell**:

```python
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# HuggingFace
HF_USERNAME  = "Basanta55"  
MODEL_REPO   = "tinyllama-alpaca-qlora"
HF_REPO_ID   = f"{HF_USERNAME}/{MODEL_REPO}"

# Model Loading
MODEL_NAME     = "unsloth/tinyllama-bnb-4bit"  # Pre-quantized for fast download
MAX_SEQ_LENGTH = 512                           # T4 safe; bump to 1024+ if needed
LOAD_IN_4BIT   = True                          # QLoRA 4-bit
DTYPE          = None                          # Auto-detect (bfloat16 on Ampere+)

# LoRA Configuration
LORA_R       = 16
LORA_ALPHA   = 16      # Usually equal to r
LORA_DROPOUT = 0.05

# Training Data
DATASET_NAME = "tatsu-lab/alpaca"
NUM_SAMPLES  = 10_000  # Use full dataset; adjust for testing
SEED         = 42

# Training Hyperparameters
EPOCHS         = 2
BATCH_SIZE     = 4         # Per-device batch size
GRAD_ACCUM     = 4         # Effective batch = 4 × 4 = 16
LEARNING_RATE  = 5e-4
WARMUP_RATIO   = 0.1       # 10% of total steps
WEIGHT_DECAY   = 0.0

# Output
OUTPUT_DIR = "./outputs"
```

To modify settings, update the configuration section in the notebook before running.

---

## 📈 Training Details

### Prompt Template

The model learns using a structured prompt format:

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

### Loss Masking

**Response-only loss masking** is applied to focus training only on the response portion:
- Instruction and input tokens: No gradients
- Response tokens: Full gradient flow
- EOS token: Included to teach the model when to stop

### Memory Optimization

- **4-bit Quantization**: Reduces model size from 4.4GB to ~1.1GB
- **LoRA**: Only 1.49% of parameters trained (~16.4M vs 1.1B)
- **Gradient Checkpointing**: Memory-efficient backpropagation
- **Packing**: Sequences packed for efficient compute
- **Unsloth**: 5-8x speedup with minimal memory overhead

---

## 🤗 Model on HuggingFace Hub

Download and use the fine-tuned model:

**Model Link**: [Basanta55/tinyllama-alpaca-qlora](https://huggingface.co/Basanta55/tinyllama-alpaca-qlora)

### Quick Load

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Basanta55/tinyllama-alpaca-qlora",
    max_seq_length=512,
    load_in_4bit=True,
)
```

### Standard Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Basanta55/tinyllama-alpaca-qlora")
model = AutoModelForCausalLM.from_pretrained("Basanta55/tinyllama-alpaca-qlora")
```

---

## 💡 Key Features

✅ **Ultra-Fast Training**: 5-8x speedup with Unsloth  
✅ **Minimal Memory**: 6-8GB VRAM sufficient (4-bit QLoRA)  
✅ **Instruction Following**: Trained on Alpaca dataset  
✅ **Production Ready**: Model pushed to HF Hub  
✅ **Reproducible**: Fixed seed and full config saved  
✅ **Well Documented**: Detailed notebook with explanations  
✅ **Inference Ready**: Easy to use for predictions  
✅ **Colab Compatible**: Run on free T4 GPU  

---

## 📊 Performance Benchmarks

### Hardware Requirements

| Hardware | VRAM | Training Time | Notes |
|----------|------|---------------|-------|
| **Colab T4** | ~6-8 GB | 15-25 min | Recommended |
| **Colab L4** | ~6 GB | 8-12 min | Faster |
| **Local RTX 3060** | ~6 GB | 10-15 min | Local GPU |
| **Local RTX 4090** | ~6 GB | 3-5 min | Enterprise |

### Inference Performance

| Device | Speed | Notes |
|--------|-------|-------|
| **CPU** | 5-10 tok/s | Very slow |
| **T4 GPU** | 50-80 tok/s | Recommended |
| **L4 GPU** | 100-150 tok/s | Fast |
| **A100 GPU** | 300+ tok/s | Enterprise |

---



### ❌ Slow Training

**Checklist**:
- ✅ GPU is being used (run `nvidia-smi`)
- ✅ Batch size is reasonable (4-8)
- ✅ No background processes using GPU
- ✅ CUDA is properly installed

**Try**: Restart runtime and reduce dataset size for testing




## 📚 Resources & References

### Papers

- 📄 [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- 📄 [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- 📄 [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)

### Documentation

- 🔗 [Unsloth GitHub](https://github.com/unslothai/unsloth)
- 🔗 [TRL Documentation](https://huggingface.co/docs/trl)
- 🔗 [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B)
- 🔗 [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
- 🔗 [HuggingFace Hub](https://huggingface.co/)

---

## 🤝 Contributing

We welcome contributions! 

**How to contribute**:
1. 🍴 Fork the repository
2. 🌿 Create feature branch (`git checkout -b feature/amazing-feature`)
3. 📝 Commit changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push branch (`git push origin feature/amazing-feature`)
5. 🔀 Open Pull Request

**Ideas**:
- Test on new instruction datasets
- Add support for other base models (Phi, Mistral, etc.)
- Create evaluation benchmarks
- Improve documentation with examples
- Add GGML export for edge devices

---
