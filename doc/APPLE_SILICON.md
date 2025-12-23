# Using LLMs on Apple Silicon (M1/M2/M3/M4)

## ✅ Great News for M4 Max Users!

Your **M4 Max** with **16-22GB unified memory** is perfect for running the Qwen2.5-7B model using **MPS (Metal Performance Shaders)** - Apple's GPU acceleration framework.

## 🚀 Quick Start for M4 Max

```bash
# Activate environment
source .venv/bin/activate

# Download and test (auto-detects MPS)
python scripts/download_and_test_llm.py
```

Or directly in code:

```python
from codenames_rl.agents.baselines import LLMSpymaster, LLMGuesser

# Auto-detects MPS on Apple Silicon
spymaster = LLMSpymaster(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # ~14GB
    # device="mps" is auto-detected, no need to specify
)

# Share model to save memory
guesser = LLMGuesser(
    model=spymaster.model,
    tokenizer=spymaster.tokenizer
)
```

## 📊 Memory Requirements on M4 Max

| Model | Unified Memory | Quality | Recommended for M4 Max |
|-------|----------------|---------|------------------------|
| Qwen2.5-0.5B | ~2GB | Poor (not production) | ✅ For testing only |
| Qwen2.5-1.5B | ~4GB | Good | ✅ Great for 16GB M4 |
| Qwen2.5-3B | ~8GB | Better | ✅ Good for 16GB+ M4 |
| Qwen2.5-7B | ~14GB | Best | ✅ Perfect for 32GB M4 Max |
| Qwen2.5-14B | ~28GB | Excellent | ⚠️  Requires 64GB+ |

**Your M4 Max with 16-22GB:** Can comfortably run the **3B model**, and the **7B model** with some memory headroom.

## 🎯 Recommended Configuration

### For M4 Max with 16GB

```python
# Use 3B for best balance
spymaster = LLMSpymaster(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    temperature=0.7,
    seed=42
)
```

### For M4 Max with 32GB+

```python
# Use full 7B model
spymaster = LLMSpymaster(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
    seed=42
)
```

## 🏎️ Performance on Apple Silicon

**Expected inference speed on M4 Max:**

| Model | Load Time | Per Move | Tokens/sec |
|-------|-----------|----------|------------|
| 1.5B | ~3s | ~0.5-1s | ~50-100 |
| 3B | ~5s | ~1-2s | ~30-50 |
| 7B | ~10s | ~2-4s | ~15-30 |

Much faster than CPU, comparable to mid-range NVIDIA GPUs!

## 🔧 Technical Details

### MPS Backend

- **Automatic Detection**: The code automatically detects MPS and uses it
- **Float32**: MPS uses float32 (not float16) for best stability
- **Unified Memory**: Shares RAM with CPU (no separate VRAM limit)
- **Neural Engine**: M4's Neural Engine may accelerate some operations

### Device Selection Logic

```python
# In baselines.py, this happens automatically:
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon ✓
else:
    device = "cpu"   # Fallback
```

## ✅ Verification

Test MPS is working:

```bash
python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
"
```

Should output:
```
MPS available: True
MPS built: True
```

## 🎮 Full Example

```python
from codenames_rl.agents.baselines import LLMSpymaster, LLMGuesser
from codenames_rl.env.core import CodenamesEnv

# Create environment
env = CodenamesEnv()

# Create agents - auto-uses MPS on M4
print("Loading model on M4 Max with MPS...")
spymaster = LLMSpymaster(
    model_name="Qwen/Qwen2.5-3B-Instruct",  # Good for 16GB
    temperature=0.7,
    seed=42
)

guesser = LLMGuesser(
    model=spymaster.model,
    tokenizer=spymaster.tokenizer,
    temperature=0.7,
    seed=42
)

# Play a game
obs = env.reset(seed=42)
clue_action = spymaster.get_clue(obs)
print(f"Spymaster clue: {clue_action.clue} ({clue_action.count})")
```

## 📝 Notes for M4 Max

1. **First run**: Downloads ~14GB (for 7B model), cached afterwards
2. **Memory**: Watch Activity Monitor - model + inference needs headroom  
3. **Temperature**: M4 Max stays cool, no thermal throttling typically
4. **Battery**: Inference is power-intensive, plug in for long sessions
5. **Speed**: Faster than CPU, ~60-80% of mid-range NVIDIA GPU speed

## 🐛 Troubleshooting

### "MPS backend out of memory"

**Solution**: Use smaller model or close other apps

```python
# Switch to 3B instead of 7B
spymaster = LLMSpymaster(model_name="Qwen/Qwen2.5-3B-Instruct")
```

### "MPS not available"

**Check**:
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

If False, ensure you have:
- macOS 12.3+
- PyTorch 2.0+ with MPS support
- M1/M2/M3/M4 chip (not Intel Mac)

### Slow performance

**Tips**:
- Close other memory-intensive apps
- Use smaller model (3B instead of 7B)
- Ensure you're not on battery saver mode
- Check Activity Monitor for swap usage

## 🎉 Summary for M4 Max Users

✅ **MPS support is built-in** - just use the agents normally  
✅ **16GB RAM**: Perfect for 3B model  
✅ **32GB+ RAM**: Perfect for 7B model  
✅ **Fast inference**: 2-4s per move with 7B model  
✅ **No external GPU needed**: Built-in acceleration  

Your M4 Max is a great platform for running these LLM agents! 🚀


