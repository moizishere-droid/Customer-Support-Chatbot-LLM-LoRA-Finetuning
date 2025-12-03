# 🤖 Support Bot GPT-2
A **LoRA fine-tuned GPT-2 model** for generating customer support responses. This project demonstrates parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation), instruction-based training, and deployment using Hugging Face Transformers and PEFT.

---

## 🚀 Highlights
* **LoRA Fine-Tuning of GPT-2**: Parameter-efficient adaptation using LoRA with only **0.24% trainable parameters**.
* **Instruction-Following Format**: Trained on instruction-response pairs for structured customer support conversations.
* **Parameter-Efficient Training**: Uses LoRA adapters instead of full model fine-tuning, training only 294,912 out of 124M+ parameters.
* **8-bit Quantization**: Model loaded in 8-bit precision for efficient inference on consumer GPUs.
* **PEFT Integration**: Leverages Hugging Face PEFT library for adapter-based fine-tuning.
* **Large-Scale Dataset**: Fine-tuned on 10,000 chatbot instruction prompts with 2,000 test examples.
* **GPU-Accelerated Training**: Mixed-precision (FP16) training with gradient accumulation for memory efficiency.

---

## 📊 Training Configuration

| Parameter | Value |
| --------- | ----- |
| Base Model | GPT-2 (124M parameters) |
| Dataset | alespalla/chatbot_instruction_prompts |
| Training Samples | 10,000 |
| Test Samples | 2,000 |
| Batch Size (per device) | 2 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 8 (2 × 4) |
| Learning Rate | 2e-4 |
| Epochs | 1 |
| Max Sequence Length | 256 tokens |
| Optimizer | AdamW (torch fused) |
| Precision | FP16 (Mixed Precision) |
| LR Scheduler | Linear |
| Max Gradient Norm | 1.0 |
| Weight Decay | 0.0 |

### LoRA Configuration
| Parameter | Value |
| --------- | ----- |
| LoRA Rank (r) | 8 |
| LoRA Alpha | 16 |
| Target Modules | c_attn |
| LoRA Dropout | 0.05 |
| Bias | none |
| Trainable Parameters | 294,912 (0.24%) |

> **Memory Efficiency**: Only 0.24% of model parameters are trainable, dramatically reducing memory requirements and training time.

---

## 🛠 Technologies & Libraries
* **Python 3.x**
* **[Transformers](https://huggingface.co/docs/transformers/index)** – Pre-trained models & fine-tuning
* **[Datasets](https://huggingface.co/docs/datasets/)** – Loading and preprocessing datasets
* **[PEFT](https://github.com/huggingface/peft)** – Parameter-Efficient Fine-Tuning with LoRA
* **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** – 8-bit quantization
* **[Evaluate](https://huggingface.co/docs/evaluate/)** – Metrics computation (accuracy)
* **PyTorch** – Model training backend with CUDA support

---

## ⚙️ Features
* **Instruction-Based Training**: Uses structured prompt format with clear instruction/response separation.
* **Reproducible Training**: Complete training arguments saved for exact experiment replication.
* **Efficient Logging**: Automatic logging every 50 steps for training monitoring.
* **Checkpoint Strategy**: Model saved every 1000 steps with a limit of 2 checkpoints to manage storage.
* **Mixed Precision Training**: FP16 precision for 2x faster training and reduced memory footprint.
* **Gradient Accumulation**: Accumulates gradients over 4 steps to simulate larger batch size with limited memory.
* **LoRA Adapters**: Small adapter weights (~300K params) can be easily shared and switched.

---

## 🎯 Training Pipeline

### Dataset Preprocessing
The model was trained on the `alespalla/chatbot_instruction_prompts` dataset with the following format:

```python
### Instruction:
{prompt}

### Response:
{response}
```

**Example:**
```
### Instruction:
How do I reset my password?

### Response:
1. Open the online customer support portal...
```

### Hardware Configuration
* **Device**: CUDA-enabled GPU
* **Quantization**: 8-bit model loading
* **Memory Optimization**: Mixed precision (FP16) + Gradient Accumulation + LoRA

### Output & Logging
* **Model Output Directory**: `/content/support-bot-gpt2`
* **Logging Strategy**: Every 50 steps
* **Evaluation Strategy**: Every 1000 steps
* **Checkpoint Storage**: Up to 2 checkpoints retained
* **Reporting**: Disabled (report_to="none")

---

## 📦 Usage

### Loading the Model with PEFT
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

### Example Output
```
❓ Question: How do I reset my password?
💬 Response: 1. Open the online customer support portal.
2. In the upper right corner of the page, click on "Reset Password"...
```

---

## 🌟 Key Learning & Focus
By building this project, I focused on:
* Understanding **LoRA fine-tuning** workflows for parameter-efficient adaptation
* Implementing **PEFT (Parameter-Efficient Fine-Tuning)** techniques
* Using **8-bit quantization** for memory-efficient inference
* Working with **instruction-following datasets** and prompt engineering
* Configuring **optimal training parameters** for GPU efficiency
* Implementing **mixed-precision training** for faster convergence
* Using **gradient accumulation** to overcome memory constraints
* Training only **0.24% of model parameters** while maintaining performance
* Managing **adapter checkpoints** for modular deployment
* Using Hugging Face **Trainer API** with PEFT integration
* Building **conversational AI systems** with instruction-following format

---

## 📌 Technical Details

### LoRA Architecture
* **Target Module**: c_attn (attention layers in GPT-2)
* **Rank (r)**: 8 - Determines the dimension of low-rank matrices
* **Alpha**: 16 - Scaling factor for LoRA weights
* **Dropout**: 0.05 - Prevents overfitting in adapter layers
* **Trainable Parameters**: Only 294,912 out of 124,734,720 total parameters

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
* Small per-device batch size (2) to fit quantized model in GPU memory
* Gradient accumulation over 4 steps to maintain training stability
* FP16 mixed precision for 50% memory reduction
* 8-bit quantization during inference
* LoRA adapters significantly reduce memory footprint

### Tokenization
* **Max Length**: 256 tokens
* **Padding**: Max length with right-side padding
* **Truncation**: Enabled for longer sequences
* **Pad Token**: Set to EOS token (50256)

---

## 🔄 Reproducibility
This project includes:
* Complete training argument serialization (`training_args.bin`)
* LoRA configuration saved with adapter weights
* Fixed random seed (42) for deterministic results
* Full configuration preservation for experiment replication
* Compatible with Hugging Face ecosystem for easy sharing

---

## 💡 Use Cases
* Customer support chatbots
* Automated response generation
* Support ticket handling
* FAQ automation
* Conversational AI assistants
* Instruction-following task completion

---

## 📈 Advantages of LoRA Fine-Tuning
* **Memory Efficient**: Train only 0.24% of parameters
* **Faster Training**: Reduced computation requirements
* **Easy Deployment**: Share small adapter files (~1-2MB) instead of full model
* **Modular**: Switch between different task-specific adapters on same base model
* **Lower Cost**: Can train on consumer GPUs with limited VRAM

---

*Training session: December 3, 2025*
