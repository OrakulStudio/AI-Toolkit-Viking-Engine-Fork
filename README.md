# AI-Toolkit - Viking Engine Fork

**High-performance LoRA training for Flux2 on RTX 4090**  
*Server-class speed on consumer hardware*

> Based on [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) - the original author's work is the foundation of everything here. All original commits preserved.
>
> Why Orakul Studio is Windows Native
Many people are used to thinking that "serious" development and AI belong on Linux. But if you look under the hood of any popular operating system, you'll see an endless series of workarounds, emulation attempts, and compromises.

We don't compromise.

Hardware Performance: The Viking Engine is optimized for Windows not out of laziness, but because it provides direct, low level access to the RTX 4090's resources without any slowdowns.

Death to workarounds: Linux is beautiful when you're browsing the web. But when it comes to working with heavyweights, asynchronous memory streaming, and real computing resources, it turns into a patchwork of patches. We value our time and studio resources more than "pretty" fonts in the terminal.

Unique Architecture: Our optimizations are the result of extensive work with the Ada Lovelace architecture. It works where it's supposed to fast, predictable, and without surprises like broken drivers or incompatible libraries.

Oracul Studio is designed for those who want to get things done, not just "administer" systems. Our base is Windows. Our goal is results. Anything else is just a waste of time.
Linux is great for servers and clean code. But when you have two hours of daylight and need to run 200 steps to rank 1280, you don't care about a "proper kernel contract." You just want it to work.

So yes, my code is written for Windows. And it runs. If anyone wants to port the Viking Engine to Linux, they're welcome, I'm not opposed. But for now, I'll stick with where it's less of a hassle and more rewarding.

*Orakul Studio - Chernihiv, Ukraine 🇺🇦*

---

## ⚠️ STRICTLY FOR TERMINAL / CLI RUNNING ONLY

If you're one of the 400+ people who cloned this repository, forget about any web UIs for this pipeline.

This code was designed, rewritten, and optimized exclusively for directly running configuration files (.yaml) via the console.

### What breaks when running via the Web UI:

1. Dynamic Alpha  DOESN'T WORK AT ALL**
* This repository implements dynamic Alpha recalculation logic for correct weight scaling (Scale = Alpha / Rank). For example, when working with high ranks (Rank 128, Rank 512, Rank 1024), the system automatically calculates a fair scale (down to Scale = 0.5000), allowing the model to deeply learn the structure and physics of the material.
* **The web UI completely ignores this logic.** Almost all web wrappers under the hood forcibly overwrite this parameter and force a fixed Alpha = 16. At high ranks, this turns training into a dud: weight changes are suppressed, gradients tend to zero, the model visually "learns" without errors, but produces default output.

2. **Asynchronous Memory Manager (Async CUDA Memory Manager)  CRASHED**
* The logic for memory retention and low-level logging is optimized for the terminal's stdout.
* Web interfaces attempt to intercept and parse the string stream for their browser consoles. At best, this leads to a crash of the backend interface due to custom security prints; at worst, to a hidden downcast of tensor precision and gradient castration, so that a casual user doesn't simply "get a memory error."

### How to use the repository correctly:
* Run strictly through the console, directly from your virtual environment.
* If your process crashes while working with high ranks and dynamic alpha, **don't look for compromises in the code; instead, increase the system swap/pagefile**. The terminal works with your hardware without censorship or hidden precision reductions.
* Here is the configuration file for running the training [Test1280.yaml](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/viking_train/Test1280.yaml)

## What This Is

The original ai-toolkit is an excellent, flexible framework. This fork takes it in one specific direction: **maximum performance on RTX 4090 (Ada Lovelace, sm_89)**.

Not about making weak hardware work. About making strong hardware **fly**.


### ⚡ Benchmark & Performance Verification (FLUX.2-Dev / RTX 4090)

| LoRA Configuration | Speed (s/it) | VRAM Memory Status | Hardware Proof |
| :--- | :--- | :--- | :--- |
| **Rank 128 (Optimized)** | **6.70s / 6.50s** | 24 GB (Zero OOM / Stable) | [Nsight / Terminal Verified] |
| **Rank 512 (Deep Gesture)** | **8.97s** | 24 GB (Double Buffered) | [Log Verified] |
| **Rank 1024 (Extreme)** | **22.45s** | 24 GB (Full 8-bit Stack Forced) | [Log Verified] |
| **Rank 1280 (Extreme)** | **65.80s** | 24 GB (Full 8-bit Stack Forced) | [Log Verified] |

---
---
## Benchmark - Flux2-dev, RTX 4090, Rank 128
[orakul_report_folder_logs](https://github.com/OrakulStudio/AI-Toolkit-Windows11/tree/main/orakul_report)

<img width="3840" height="2160" alt="10" src="https://github.com/user-attachments/assets/b9e5c210-ac1a-4d96-b142-d0611de913f1" />
<img width="3840" height="2160" alt="3" src="https://github.com/user-attachments/assets/c107b7e7-ed27-485b-a449-28507ba0afef" />
<img width="3840" height="2160" alt="8" src="https://github.com/user-attachments/assets/ac55b44c-bb40-404c-99f7-ac6ba547469c" />

[orakul_report.txt](https://github.com/OrakulStudio/AI-Toolkit-Windows11/blob/main/orakul_report/orakul_report.txt)


## Benchmark - Flux2-dev, RTX 4090, Rank 512

| Version | Speed | What Changed |
|---------|-------|-------------|
| Baseline (original) | 179 s/it | — |
| Viking v1 | 37 s/it | Double-buffer async CUDA |
| Viking v2 | 14 s/it | + bf16 weight forcing |
| Oracle-60 | 8.7 s/it | + Hardware FP8 + CPU prequant |
| **LEGEND** | **7.3 s/it** | **+ Full 8-bit stack (AdamW 8-bit)** |

**24.5× faster than baseline. Same hardware. Zero OOM at rank 1280 (7.8B trainable params).**

---

## Modified Files

### [toolkit/manager_modules.pyd](https://github.com/OrakulStudio/AI-Toolkit-Windows11/blob/main/toolkit/memory_management/manager_modules.pyd) — Viking Engine
**Double-buffered async weight streaming.**

While GPU computes layer N, weights for layer N+1 transfer in a parallel CUDA stream. Transfer disappears from the profiler entirely.

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

Also: CPU pinned memory for direct DMA from DRAM without CPU cache copy.

---

### [toolkit/quantize.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/util/quantize.py) — Protocol Oracle-60
**Native FP8 (E5M2) for Ada Lovelace + CPU pre-quantization.**

RTX 4090 has native FP8 Tensor Cores. This activates them.  
CPU pre-quantizes transformer blocks before GPU load  PCIe bus freed.

```python
"bf8": Float8WeightOnlyConfig(weight_dtype=torch.float8_e5m2)
```

```
>>> [ORACLE-60] BF8 NATIVE (E5M2) DETECTED - CPU PRE-QUANT MODE
```

**Why E5M2:** preserves dynamic range like BF16. No "muddy faces" from aggressive quantization. Skin texture survives.

---

### [toolkit/lora_special.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/lora_special.py)  Viking Override
**LoRA matrices born in bfloat16, not converted later.**

```python
# === VIKING OVERRIDE: ЖЕСТКИЙ BFLOAT16 ДЛЯ ВЕСОВ ===
dtype = torch.bfloat16
```

All Linear and Conv2d LoRA layers initialized directly in bf16. Half the memory at birth. PCIe transfer halved from step one.

**Also fixed:** Alpha/scale calculation for rank 1024.  
Original code skipped `.alpha` keys during save, causing **64× signal drop** at rank 1024 with alpha 64. Fixed and locked:

```python
# === ORACLE STANDART PASS (FIXED) ===
alpha_val = alpha if alpha is not None and alpha != 0 else lora_dim
self.scale = alpha_val / self.lora_dim
```

---

### [toolkit/network_mixins.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/network_mixins.py) — Oracle-60 Alpha Fix
**The alpha skip bug that corrupted high-rank training.**

Lines that skipped `.alpha` keys for non-LoKR networks are permanently commented with explanation. The `lora_special.py` autopilot handles scale correctly now.

```python
# === [ORACLE-60] BLOCK: ЗАЩИТА ОТ МЫЛА ===
# These lines skipped Alpha for all types except LoKR.
# At Rank 1024, signal dropped 64x (Scale 16/1024 instead of 64/1024).
# KEEP DISABLED  lora_special.py autopilot handles this correctly.
```

---

### [toolkit/style.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/style.py) — Perceptual Loss (Rewritten)
**VGG19 perceptual loss without the VRAM leak.**

Original had a critical bug: VGG19 was computing gradients for itself during perceptual loss calculation — wasting gigabytes of VRAM on a frozen reference network.

Fixed:
```python
# Freeze VGG19 completely — it's a reference, not a trainee
for param in cnn.parameters():
    param.requires_grad = False
```

Also: Gram matrix calculation replaced with hardware `torch.bmm` (Batch Matrix Multiply) — faster, cleaner, no nested function overhead.

**Use perceptual loss for:** Aivazovsky, watercolor, oil painting, charcoal, any artistic style where brushstroke texture matters more than pixel accuracy.

**Use MSE for:** portraits, photorealism, identity training.

---

### [toolkit/timer.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/timer.py) — Timer Silenced
**CPU overhead from time polling eliminated.**

Original timer called `time.time()` on every micro-step of the pipeline — `predict_unet`, `backward`, `optimizer_step`, etc. Parasitic CPU load, log spam, potential micro-freeze points on PCIe bus under heavy load.

```python
def start(self, timer_name):
    return  # CPU freed

def stop(self, timer_name):
    return  # bus cleared

def print(self):
    for hook in self._after_print_hooks:
        hook({})  # engine hooks get empty dict, nothing breaks
    return
```

Result: cleaner logs, lower CPU load, stable PCIe bus under sustained training.

---

### [jobs/process/BaseSDTrainProcess.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/jobs/process/BaseSDTrainProcess.py) — bf16 Forcing
**Network forced to bfloat16 before training starts.**

```python
# Viking method — before network.apply_to()
# todo switch everything to proper mixed precision like this  ← ostris left this todo
self.network.force_to(self.device_torch, dtype=torch.bfloat16)
```

The `# todo` comment was already in the original source. We read it and implemented it.

---

## Two Training Modes

### Mode 1: Photorealism (Maximum Speed)
For portraits, faces, identity training. Pixel-accurate, fast.

```yaml
content_or_style: balanced
loss_type: mse
```

### Mode 2: Art Styles (Perceptual)
For Aivazovsky, watercolor, oil, charcoal. Learns brushstroke, not pixels.  
Uses ~1-2 GB more VRAM, ~20-30% slower, higher GPU voltage.

```yaml
content_or_style: style
loss_type: perceptual
```

Switch between modes by commenting/uncommenting — no config rewrite needed.

---

## Recommended Config

```yaml
network:
  type: lora
  linear: 1280          # rank 1280 = 7.8B trainable params
  linear_alpha: 64
  conv: 32
  conv_alpha: 64
  lokr_full_rank: true
  lokr_factor: -1

model:
  qtype: bf8            # Oracle-60 native FP8
  quantize_te: true
  qtype_te: bf8
  layer_offloading: true
  layer_offloading_transformer_percent: 0.91
```

---

## Launch

Web UI works but **sampling is disabled for maximum speed**.  
For real performance — terminal only:

```bash
# Activate environment
source venv/Scripts/activate

# Run training with full log capture
python run.py viking_train/your_config.yaml 2>&1 | tee "log_training.txt"
```

Configs go in `viking_train/` folder.

---

## Installation & Requirements

- RTX 4090 (24 GB) — optimized for Ada Lovelace sm_89
- 64+ GB RAM recommended (128 GB for rank 1280)
- Python 3.10+ / Linux or Windows 11

```bash
git clone https://github.com/OrakulStudio/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

If you encounter numpy.dtype errors or CUDA/Triton warnings, use the following commands to ensure your environment is set up correctly for high-performance training:

Fix NumPy/SciPy version conflicts:
pip install "numpy<2.0.0" scipy --force-reinstall
---

Configs go in viking_train/ folder.

What Ostris Built
This fork exists because ostris built something worth building on.
ostris/ai-toolkit is the most widely used open-source LoRA training framework. Thousands of people use it daily. It's clean, flexible, actively maintained.

All original commits preserved. Author credited. The original README can be found in README_OSTRIS.md.


---

## Links

- 🐙 **Original:** [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- 🐙 **This fork:** [github.com/OrakulStudio](https://github.com/OrakulStudio)
- 🤗 [huggingface.co/OrakulStorm](https://huggingface.co/OrakulStorm)
- 🎨 [civitai.com/user/orakul_storm](https://civitai.com/user/orakul_storm)

---


*The smell of the iron is stable. 🦊⚡*

*Chernihiv, Ukraine 🇺🇦 · Orakul Studio · 2026*
