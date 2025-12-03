# 🤖 Support Bot GPT-2
A **LoRA fine-tuned GPT-2 model** for generating customer support responses. This project demonstrates parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation), efficient training configuration, and deployment using Hugging Face Transformers and PEFT.

---

## 🚀 Highlights
* **LoRA Fine-Tuning of GPT-2**: Parameter-efficient adaptation using LoRA for customer support conversation generation.
* **Parameter-Efficient Training**: Uses LoRA adapters instead of full model fine-tuning, reducing memory requirements by 90%.
* **8-bit Quantization**: Model loaded in 8-bit precision for efficient inference on consumer GPUs.
* **PEFT Integration**: Leverages Hugging Face PEFT library for adapter-based fine-tuning.
* **GPU-Accelerated Training**: Leverages CUDA with fused AdamW optimizer for faster convergence.
* **Checkpoint Management**: Automatic model saving every 1000 steps with checkpoint limits for storage efficiency.

---

## 📊 Training Configuration

| Parameter | Value |
| --------- | ----- |
| Base Model | GPT-2 |
| Batch Size (per device) | 2 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 8 (2 × 4) |
| Learning Rate | 2e-4 |
| Epochs | 1 |
| Optimizer | AdamW (torch fused) |
| Precision | FP16 (Mixed Precision) |
| LR Scheduler | Linear |
| Max Gradient Norm | 1.0 |
| Weight Decay | 0.0 |

> Optimized for single-GPU training with gradient accumulation to simulate larger batch sizes while maintaining memory efficiency.

---

## 🛠 Technologies & Libraries
* **Python 3.x**
* **[Transformers](https://huggingface.co/docs/transformers/index)** – Pre-trained models & fine-tuning
* **[Datasets](https://huggingface.co/docs/datasets/)** – Loading and preprocessing datasets
* **PyTorch** – Model training backend with CUDA support
* **[PEFT](https://github.com/huggingface/peft)** – Parameter-Efficient Fine-Tuning with LoRA
* **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** – 8-bit quantization

---

## ⚙️ Features
* **Reproducible Training**: Complete training arguments saved for exact experiment replication.
* **Efficient Logging**: Automatic logging every 50 steps for training monitoring.
* **Checkpoint Strategy**: Model saved every 1000 steps with a limit of 2 checkpoints to manage storage.
* **Mixed Precision Training**: FP16 precision for 2x faster training and reduced memory footprint.
* **Gradient Accumulation**: Accumulates gradients over 4 steps to simulate larger batch size with limited memory.
* **Memory Optimization**: Small batch size (2) with gradient accumulation for GPU memory efficiency.

---

## 🎯 Training Pipeline

### Hardware Configuration
* **Device**: CUDA-enabled GPU
* **Distributed Type**: Single GPU (no multi-GPU setup)
* **Memory Optimization**: Mixed precision (FP16) + Gradient Accumulation

### Output & Logging
* **Model Output Directory**: `/content/support-bot-gpt2`
* **Logging Strategy**: Every 50 steps
* **Evaluation Strategy**: Every 1000 steps
* **Checkpoint Storage**: Up to 2 checkpoints retained (saves storage space)
* **Reporting**: Disabled (report_to="none")

---

## 📦 Usage

### Loading the Model
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load fine-tuned model
MODEL_PATH = "support-bot-model"  # path where you saved LoRA model
BASE_MODEL = "gpt2"               # base model used for fine-tuning

print("📥 Loading model for inference...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

# Function to generate response
def generate_response(question, max_new_tokens=150, temperature=0.7, top_p=0.9):
    prompt = f"""Below is an instruction that describes a task, paired with an input. Write a response that appropriately completes the request.

### Instruction:
Answer this customer support question professionally and helpfully.

### Input:
{question}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("### Response:")[-1].strip()
    return response

# Test multiple questions
test_questions = [
    "How do I reset my password?",
    "What's your warranty policy?",
    "Can I get express shipping?",
    "Do you have a mobile app?",
]

print("\n" + "="*60)
print("🤖 Customer Support Chatbot Responses")
print("="*60 + "\n")

for q in test_questions:
    answer = generate_response(q)
    print(f"❓ Question: {q}")
    print(f"💬 Response: {answer}\n")
    print("-" * 60 + "\n")
```

### Resuming Training
```python
from transformers import Trainer, TrainingArguments
import torch

# Load saved training arguments
training_args = torch.load("training_args.bin")

# Resume training from checkpoint
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train(resume_from_checkpoint=True)
```

---

## 🌟 Key Learning & Focus
By building this project, I focused on:
* Understanding **LoRA fine-tuning** workflows for parameter-efficient adaptation
* Implementing **PEFT (Parameter-Efficient Fine-Tuning)** techniques
* Using **8-bit quantization** for memory-efficient inference
* Configuring **optimal training parameters** for GPU efficiency
* Implementing **mixed-precision training** for faster convergence
* Using **gradient accumulation** to overcome memory constraints
* Managing **adapter checkpoints** for modular deployment
* Using Hugging Face **Trainer API** with PEFT integration
* **Reproducibility** through comprehensive configuration preservation
* Building **conversational AI systems** with instruction-following format

---

## 📌 Technical Details

### Optimizer Configuration
* **Type**: AdamW with fused operations
* **Beta1**: 0.9
* **Beta2**: 0.999
* **Epsilon**: 1e-8
* **Weight Decay**: 0.0

### Training Strategy
* **Epochs**: 1 (fast iteration for experimentation)
* **Batch Size**: 2 per device
* **Gradient Accumulation**: 4 steps (effective batch size of 8)
* **Logging**: Every 50 steps
* **Saving**: Every 1000 steps
* **Evaluation**: Every 1000 steps
* **Max Checkpoints**: 2 (older checkpoints automatically removed)

### Memory Optimization
* Small per-device batch size (2) to fit large model in GPU memory
* Gradient accumulation over 4 steps to maintain training stability
* FP16 mixed precision for 50% memory reduction
* Efficient checkpoint management with max 2 saved models

### Stability Features
* Gradient clipping at norm 1.0
* Linear learning rate scheduler
* Fixed random seed (seed: 42)

---

## 🔄 Reproducibility
This project includes:
* Complete training argument serialization (`training_args.bin`)
* Fixed random seed for deterministic results
* Full configuration preservation for experiment replication
* Compatible with Hugging Face ecosystem for easy sharing

---

## 💡 Use Cases
* Customer support chatbots
* Automated response generation
* Support ticket handling
* FAQ automation
* Conversational AI assistants
