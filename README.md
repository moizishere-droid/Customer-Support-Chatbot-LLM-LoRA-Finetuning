# 🤖 Support Bot GPT-2
A **task-specific fine-tuned GPT-2 model** for generating customer support responses. This project demonstrates full LLM fine-tuning, efficient training configuration, and deployment using Hugging Face Transformers.

---

## 🚀 Highlights
* **Full Fine-Tuning of GPT-2**: Adapted a pre-trained language model for customer support conversation generation.
* **Task-Specific Training**: Fine-tuned on custom support dialogue dataset for contextually relevant responses.
* **Optimized Training Pipeline**: Configured with mixed-precision (FP16) training and gradient accumulation for efficient GPU usage.
* **Production-Ready Configuration**: Complete training arguments preserved for reproducibility and model deployment.
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
* **[Accelerate](https://huggingface.co/docs/accelerate/)** – Distributed training utilities

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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load your fine-tuned model
OUTPUT_DIR = "/content/support-bot-gpt2"  # Folder where your model was saved

print("📥 Loading fine-tuned model for inference...")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, device_map="auto", torch_dtype=torch.float16)
model.eval()

# Function for generating a response
def generate_response(prompt, max_length=200, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=200,
            temperature=0.8,  # slightly higher randomness
            top_p=0.95,       # nucleus sampling
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2  # discourages repetition
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove prompt from output
    return response[len(prompt):].strip()

# Example usage
prompt = "Answer this customer support question professionally and helpfully:\nHow do I return an item?"
response = generate_response(prompt)
print("🟢 Response:\n", response)
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
* Understanding **full LLM fine-tuning** workflows for generative tasks
* Configuring **optimal training parameters** for GPU efficiency
* Implementing **mixed-precision training** for faster convergence
* Using **gradient accumulation** to overcome memory constraints
* Managing **checkpoints and model versioning** in production
* Using Hugging Face **Trainer API** with custom training arguments
* **Reproducibility** through comprehensive configuration preservation
* Building **conversational AI systems** for practical applications

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

---

*Training session: December 3, 2025*
