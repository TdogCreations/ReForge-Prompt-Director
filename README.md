# ReForge Prompt Director

**ReForge Prompt Director** is a modular prompt orchestration suite for  
**Stable Diffusion WebUI ReForge**.

It combines **JoyCaption Ultra**, **WD14 Tagger**, and **QuickShot Prompt Director**
into a single, batch-safe system.

Each module can run **independently** or **together**, and all are fully compatible
with ReForge’s batch and prompt-matrix execution model.

---

## 🧠 System Overview (Important)

ReForge Prompt Director is **not a single model**.

It is a **controller layer** that coordinates multiple systems:

| Module | Purpose |
|------|--------|
| **JoyCaption Ultra** | Vision-based captioning + prompt rewrite |
| **WD14 Tagger** | Image → tag inference (ONNX) |
| **QuickShot** | Deterministic prompt steering (age, camera, time, lighting) |

These modules **share state safely** during batch execution.

---

## 🔁 Execution Order (How it actually works)

For each batch / prompt index:

1. **QuickShot** modifies the prompt (no image required)
2. **WD14 Tagger** selects the correct image for the batch index
3. **WD14 tags** are injected into the prompt
4. **JoyCaption** (optional) captions the *same image*
5. Final prompt is sent to the sampler

This prevents:
- repeated captions
- wrong images being analyzed
- prompt/image desync during batches

---

---

## 📥 Model Downloads & Requirements

ReForge Prompt Director does **not** bundle large models.
You must download the required models manually.

This is intentional to keep the extension lightweight and transparent.

---

## 🏷️ WD14 Tagger Models (Required for WD14)

### ✅ Recommended (Best Quality)

**WD EVA02 Large Tagger v3**

- ONNX:
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/wd-eva02-large-tagger-v3.onnx
- CSV:
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/wd-eva02-large-tagger-v3.csv

**Install to:**
stable-diffusion-webui-reForge/
└─ models/
└─ wd14/
├─ wd-eva02-large-tagger-v3.onnx
└─ wd-eva02-large-tagger-v3.csv

yaml
Copy code

---

### 🟡 Alternative (Lower VRAM / Older GPUs)

**WD EVA02 Large Tagger v2**

- ONNX:
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v2/resolve/main/wd-eva02-large-tagger-v2.onnx
- CSV:
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v2/resolve/main/wd-eva02-large-tagger-v2.csv

This version is slightly less accurate but still fully supported.

---

### ⚙️ Configure Paths

After downloading:

1. Open **WebUI → Settings → WD14 Tagger**
2. Set:
   - WD14 Model Path → `.onnx`
   - WD14 Tags Path → `.csv`
3. Click **Apply settings**
4. Restart WebUI

Paths are stored in WebUI config, **not** in this extension.

---

## 🧠 JoyCaption Models (Required for JoyCaption)

JoyCaption requires a **vision-capable caption model**.

Supported families include:
- JoyCaption v1.9.x
- Qwen-VL compatible caption models

**Install to:**
stable-diffusion-webui-reForge/
└─ models/
└─ joycaption/

yaml
Copy code

⚠️ Models are **not auto-downloaded**.

If JoyCaption fails to load, the model is missing or incorrectly placed.

---

## 💾 Minimum System Requirements

### ✅ GPU VRAM

- **Minimum:** 11 GB VRAM (Low VRAM Mode)
- **Recommended:** 16 GB+ VRAM

ReForge Prompt Director is designed to work efficiently:

- WD14 uses ONNX inference (lightweight)
- JoyCaption supports quantized models
- Models are loaded only when needed
- No duplicate image processing per batch

On **11 GB GPUs**, most workflows work correctly when:
- Low VRAM mode is enabled
- Large SDXL checkpoints are avoided
- One vision model is loaded at a time

---

### 🧮 CPU / RAM

- CPU: Any modern x64 CPU
- System RAM: 16 GB minimum (32 GB recommended for large batches)

---

### 🧠 What “Low VRAM Mode” means

Low VRAM mode:
- Avoids holding multiple large models in memory
- Loads vision models only when used
- Releases intermediate buffers aggressively

This allows **image captioning + tagging** to run even on mid-range GPUs.

---

## ⚠️ Common Mistakes

- WD14 does nothing → ONNX/CSV paths not set
- JoyCaption fails → model not installed
- Batch captions repeat → restart WebUI after install
- OOM errors → enable Low VRAM mode or reduce batch size

---

### 3️⃣ Stable Diffusion Model

Any SD / SDXL model supported by ReForge works.

This extension **does not modify samplers or schedulers**.

---

## 🚀 Installation

1. Open **Stable Diffusion WebUI ReForge**
2. Go to **Extensions → Install from URL**
3. Paste:
https://github.com/TdogCreations/ReForge-Prompt-Director

yaml
Copy code
4. Click **Install**
5. Restart WebUI

---

## ⚙️ Usage Guide

### JoyCaption Ultra
- Enable in **JoyCaption tab**
- Select caption model
- Optional rewrite rules
- Batch-safe by design

### WD14 Tagger
- Enable in **WD14 Tagger tab**
- Select image source:
- Folder
- Pixiv
- Reference image
- Supports batch indexing modes:
- Increment
- Random
- Fixed

### QuickShot Prompt Director
- Enable in **QuickShot tab**
- Controls:
- Age group
- Time of day
- Camera angle
- Lighting direction
- Works with or without images

---

## 🧪 Batch & Prompt Matrix Support

Fully compatible with:
- ReForge batch execution
- Prompt matrix
- Dynamic prompts
- Multi-image queues

State is stored on the **Script instance**, not per-prompt,
to avoid ReForge’s batch recreation behavior.

---

## 🔐 Privacy & Safety

- No telemetry
- No data collection
- Pixiv PHPSESSID is stored **only** in WebUI settings
- No credentials are written to this repository

---

## 🧩 Modular Design

You may use:
- Only JoyCaption
- Only WD14
- Only QuickShot
- Any combination

Disabling one module does **not** break the others.

---

## 🛠️ Troubleshooting

- If WD14 does nothing → check ONNX/CSV paths
- If JoyCaption does not load → model missing
- If captions repeat → ensure ReForge ≥ latest
- If batching breaks → restart WebUI after install

---

