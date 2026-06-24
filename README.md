---
library_name: transformers
tags:
- recipe
- cooking
- vietnamese
- lora
- sft
license: apache-2.0
datasets:
- AkashPS11/recipes_data_food.com
language:
- vi
- en
base_model: Qwen/Qwen2.5-7B-Instruct
pipeline_tag: text-generation
---

# Qwen2.5-7B-Chef-VN

Qwen2.5-7B-Chef-VN is a fine-tuned large language model specialized in the culinary domain. Acting as an interactive "Master Chef", the model is optimized to provide highly detailed, step-by-step cooking instructions, accurate ingredient measurements, and tailored culinary advice primarily in Vietnamese, while maintaining strong English capabilities.

## Model Details

### Model Description

This model was trained using Supervised Fine-Tuning (SFT) via QLoRA (4-bit quantization) on top of the `Qwen/Qwen2.5-7B-Instruct` base architecture. The primary training data is derived from a filtered and structured translation of the `AkashPS11/recipes_data_food.com` dataset, converted into a standardized conversational ChatML format to facilitate natural multi-turn culinary assistance.

- **Developed by:** NotIsora
- **Model type:** Causal Language Model (Fine-tuned via QLoRA)
- **Language(s) (NLP):** Vietnamese (vi), English (en)
- **License:** Apache 2.0
- **Finetuned from model:** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Hugging Face Model:** [NotIsora/Qwen2.5-7B-Chef-VN](https://huggingface.co/NotIsora/Qwen2.5-7B-Chef-VN)
- **Hugging Face Dataset:** [AkashPS11/recipes_data_food.com](https://huggingface.co/datasets/AkashPS11/recipes_data_food.com)

## Uses

### Direct Use

The model can be deployed directly as a virtual chef or kitchen assistant. Users can inquire about specific dish recipes or supply a list of raw ingredients to receive comprehensive cooking roadmaps containing:
- Formatted ingredient checklists with exact quantities.
- Chronological, clear preparation and cooking milestones.
- Culinary pro-tips, techniques, and potential pitfalls to avoid.

### Out-of-Scope Use

- Medical, strict macro-nutritional, or therapeutic dietary planning.
- Generation of automated content entirely disconnected from culinary arts, food science, or recipe curation (general capabilities remain but domain specialization is the target).

## How to Get Started with the Model

You can run inference using the full combined architecture (Base Model + LoRA weights) with the following optimized pipeline setup:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LORA_WEIGHTS_ID = "NotIsora/Qwen2.5-7B-Chef-VN" 

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS_ID)

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)

# Merge Adapter at runtime
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_ID)
model.eval()

# Conversational Input Setup
test_messages = [
    {
        "role": "system", 
        "content": "You are a master chef. Your task is to provide extremely comprehensive, lengthy, and detailed culinary guides. Include exact techniques, step-by-step execution, and deep explanations."
    },
    {
        "role": "user", 
        "content": "Hãy hướng dẫn tôi chi tiết cách làm món khoai tây nghiền mượt và ngon nhất."
    }
]

test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

# Max new tokens is set to 2048 to guarantee long-form, highly exhaustive responses
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=True,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

## Training Details

### Training Data
The model was trained on a processed subset of [AkashPS11/recipes_data_food.com](https://huggingface.co/datasets/AkashPS11/recipes_data_food.com). The source raw data rows were dynamically unrolled, cleaned of syntax fragments, combined via structural mapping vectors, and injected into specialized system/user/assistant token blocks.

### Training Procedure

#### Training Hyperparameters
- **Training Precision:** bf16 mixed precision (QLoRA 4-bit base)
- **Epochs:** 6
- **Max Sequence Length:** 1024 tokens
- **Per-device Batch Size:** 2
- **Gradient Accumulation Steps:** 4
- **Effective Global Batch Size:** 8
- **Optimizer:** paged_adamw_8bit
- **Learning Rate (Max):** 5e-5
- **Learning Rate Scheduler:** Cosine Convergence
- **Warmup Ratio:** 0.1
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Technical Specifications

### Compute Infrastructure
#### Hardware
- **GPU:** 1x NVIDIA L4 Tensor Core GPU (or stable T4 environments)

#### Training Framework: 
PyTorch, Transformers, PEFT, TRL
