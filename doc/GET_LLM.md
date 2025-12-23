# Getting the Real LLM (Qwen2.5-7B-Instruct)

## Quick Start

### Option 1: Automated Download Script (Recommended)

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run the download and test script
python scripts/download_and_test_llm.py

# Or force CPU usage (slow, not recommended)
python scripts/download_and_test_llm.py --cpu
```

This will:
1. Check your system (GPU availability, VRAM)
2. Download Qwen2.5-7B-Instruct (~14GB, cached in `~/.cache/huggingface/`)
3. Test both Spymaster and Guesser agents
4. Validate the outputs

### Option 2: Manual Python Code

```python
from codenames_rl.agents.baselines import LLMSpymaster, LLMGuesser

# Download and load the full 7B model
spymaster = LLMSpymaster(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # Full 7B model
    device="cuda",      # or "cpu" (slow)
    temperature=0.7,
    seed=42
)

# Share the model to save memory
guesser = LLMGuesser(
    model=spymaster.model,
    tokenizer=spymaster.tokenizer,
    device="cuda",
    temperature=0.7,
    seed=42
)

# Use them with your game environment
action = spymaster.get_clue(observation)
```

---

## Requirements

### Hardware Requirements

**For Qwen2.5-7B-Instruct:**

| Configuration | VRAM | Speed | Recommended |
|---------------|------|-------|-------------|
| FP16 (GPU) | ~14GB | Fast | ✅ Yes (RTX 3090, A100, etc.) |
| FP32 (GPU) | ~28GB | Fast | Only if you have the VRAM |
| CPU | 0GB (uses RAM) | Very Slow | ❌ Not for production |

**Alternatives if you don't have 14GB GPU:**

| Model | VRAM | Quality |
|-------|------|---------|
| Qwen2.5-1.5B-Instruct | ~3GB | Good |
| Qwen2.5-3B-Instruct | ~6GB | Better |
| Qwen2.5-7B-Instruct | ~14GB | Best |

### Software Requirements

```bash
# Already in pyproject.toml, but verify:
pip install torch>=2.0.0
pip install transformers>=4.46.0
pip install accelerate>=1.0.0
```

---

## Download Process

### First Time (Downloads Model)

The model will automatically download from HuggingFace on first use:

```bash
# This downloads ~14GB to ~/.cache/huggingface/hub/
python scripts/download_and_test_llm.py
```

**What gets downloaded:**
- Model weights: ~13.5GB
- Tokenizer config: ~1MB
- Model config: ~1KB

**Download location:**
```
~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/
```

### Subsequent Uses (Fast)

After the first download, the model loads from cache (no download needed):

```python
# Loads from cache - fast!
spymaster = LLMSpymaster(model_name="Qwen/Qwen2.5-7B-Instruct")
```

---

## Using Different Models

### Smaller Models (Less Memory)

```python
# 1.5B model - only ~3GB VRAM
spymaster = LLMSpymaster(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda"
)

# 3B model - ~6GB VRAM  
spymaster = LLMSpymaster(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    device="cuda"
)
```

### Other Models (Advanced)

You can use any HuggingFace causal LM model:

```python
spymaster = LLMSpymaster(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",  # Alternative
    device="cuda"
)
```

**Note:** Prompts are optimized for Qwen2.5 chat format. Other models may need prompt adjustments.

---

## Troubleshooting

### "CUDA out of memory"

**Problem:** GPU doesn't have enough VRAM for the model.

**Solutions:**

1. Use a smaller model:
   ```python
   spymaster = LLMSpymaster(model_name="Qwen/Qwen2.5-1.5B-Instruct")
   ```

2. Use CPU (slow):
   ```python
   spymaster = LLMSpymaster(device="cpu")
   ```

3. Use quantization (advanced):
   ```python
   # Install bitsandbytes first: pip install bitsandbytes
   from transformers import BitsAndBytesConfig
   
   # 4-bit quantization - reduces memory by ~4x
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
   )
   
   # Then modify baselines.py to pass quantization_config to from_pretrained()
   ```

### "No module named 'torch'"

**Problem:** PyTorch not installed.

**Solution:**
```bash
source .venv/bin/activate
pip install torch transformers accelerate
```

### "Model download is slow"

**Problem:** HuggingFace servers or network speed.

**Solutions:**

1. Use a mirror (if in China):
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. Pre-download using CLI:
   ```bash
   huggingface-cli download Qwen/Qwen2.5-7B-Instruct
   ```

3. Wait - first download is ~14GB, subsequent loads are instant.

### "ModuleNotFoundError: No module named 'torch'"

Your virtual environment isn't activated:

```bash
source .venv/bin/activate
```

---

## Performance Benchmarks

Based on typical hardware:

| Setup | Load Time | Inference Speed |
|-------|-----------|-----------------|
| RTX 3090 (24GB) | ~10s | ~1-2s per move |
| RTX 4090 (24GB) | ~8s | ~0.5-1s per move |
| A100 (40GB) | ~5s | ~0.3-0.5s per move |
| CPU (32GB RAM) | ~30s | ~10-30s per move |

---

## Verifying Installation

```bash
# Quick test
source .venv/bin/activate
python -c "
from codenames_rl.agents.baselines import LLMSpymaster
import torch
print(f'✓ Torch: {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print('✓ LLMSpymaster available')
"
```

---

## Next Steps

Once the model is downloaded:

1. **Test it**: Run `python scripts/download_and_test_llm.py`
2. **Integrate**: Use in your evaluation harness
3. **Benchmark**: Compare against embeddings baseline
4. **Fine-tune**: Use as base model for SFT/DPO training

---

## Summary

**To get the real 7B LLM:**

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run download script (or just use the agents directly)
python scripts/download_and_test_llm.py

# 3. Use in your code
python -c "
from codenames_rl.agents.baselines import LLMSpymaster
spymaster = LLMSpymaster(model_name='Qwen/Qwen2.5-7B-Instruct')
print('✓ Ready to use!')
"
```

The first run downloads ~14GB. All subsequent runs load from cache instantly.


