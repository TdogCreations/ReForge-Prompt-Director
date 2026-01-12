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

## 📦 Required Models

### 1️⃣ JoyCaption Models (Required for JoyCaption)

You must install **one JoyCaption-compatible vision model**.

Supported:
- JoyCaption v1.9.x family
- Qwen-VL compatible caption models

Place models in:
stable-diffusion-webui-reForge/
└─ models/
└─ joycaption/

yaml
Copy code

⚠️ JoyCaption will **not auto-download models**.

---

### 2️⃣ WD14 Tagger Models (Required for WD14)

You must provide:

- **WD14 ONNX model**
- **WD14 tags CSV**

Recommended:
- `wd-v1-4-vit-tagger-v2.onnx`
- `tags.csv`

Set paths in:
Settings → WD14 Tagger

yaml
Copy code

Paths are stored in **WebUI settings**, not in this extension.

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
