# ReForge Prompt Director
A modular **auto-prompt + image-to-tags + prompt steering** suite for **Stable Diffusion WebUI ReForge**.

Turn reference images into usable prompts fast, keep **batches in-sync**, and optionally do **character swaps** + **term switches** (ex: male ↔ female) without your prompts breaking in ReForge batch / prompt-matrix.

---

## ✨ What this extension does (in one sentence)

**It automatically generates and injects prompts into your workflow using images**, so you spend less time writing prompts and more time generating.

### Typical uses
- **Grab an image → get a clean prompt** (caption + tags) you can generate from
- **Batch folders** of reference images safely (no “wrong image” or repeated caption bugs)
- **Character swap**: use Image 2 to swap/replace character traits into Image 1’s scene
- **Prompt steering** (age / camera angle / time / lighting) for consistent sets

---

## ✅ Key Features
- **Batch-safe by design** (works with ReForge batch, prompt matrix, dynamic prompts)
- **JoyCaption Ultra**
  - image captioning → SD-style prompt output
  - optional rewrite rules + constraint enforcement
  - optional post-switches (ex: term swaps)
- **WD14 Tagger**
  - ONNX inference → danbooru-style tag payloads
  - folder + Pixiv cache + single image sources
  - fusion modes (combine / replace) for **character swapping**
- **QuickShot Prompt Director**
  - deterministic prompt steering for consistent sets
  - age/time/view/camera/light controls
- Modules can run **independently** or **together**

---

## 🧠 How it works (execution order)

For each prompt index in a batch:

1. **QuickShot** modifies prompt text (no image needed)
2. **WD14** selects the correct image for that index
3. **WD14 tags** inject into the prompt
4. **JoyCaption** captions the *same image* (optional)
5. Final prompt goes to the sampler

This prevents:
- repeated captions
- prompt/image desync in batch runs
- the “first image gets reused” batch bug

---

## 📥 Required Model Downloads

This extension does **not** bundle large models. You install them manually.

### 1) WD14 Tagger (Required for WD14)

#### ✅ Recommended (best quality)
**WD EVA02 Large Tagger v3**
- ONNX:  
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/wd-eva02-large-tagger-v3.onnx
- CSV:  
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/wd-eva02-large-tagger-v3.csv

#### 🟡 Alternative (older / slightly lighter)
**WD EVA02 Large Tagger v2**
- ONNX:  
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v2/resolve/main/wd-eva02-large-tagger-v2.onnx
- CSV:  
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v2/resolve/main/wd-eva02-large-tagger-v2.csv

#### Install WD14 files here
stable-diffusion-webui-reForge/
└─ models/
└─ wd14/
├─ wd-eva02-large-tagger-v3.onnx
└─ wd-eva02-large-tagger-v3.csv

---

### 2) JoyCaption Model (Required for JoyCaption)

✅ **llama-joycaption-beta-one**  
Hugging Face:  
https://huggingface.co/tsunemoto/llama-joycaption-beta-one

> You may need to be logged into Hugging Face to download.

Install here:
stable-diffusion-webui-reForge/
└─ models/
└─ LLM/
└─ llama-joycaption-beta-one/
├─ config.json
├─ generation_config.json
├─ model.safetensors
├─ tokenizer.json
├─ tokenizer_config.json
└─ special_tokens_map.json

**Do not rename the folder.**  
**Do not place it inside the extension folder.**

---

## ⚙️ Setup (Paths in WebUI)

### WD14 Paths
1. WebUI → **Settings → WD14 Tagger**
2. Set:
   - **WD14 Model Path** → `.onnx`
   - **WD14 Tags Path** → `.csv`
3. Apply settings → restart WebUI

### JoyCaption Path
1. WebUI → **Settings → JoyCaption**
2. Set **JoyCaption Model Path** to your local model folder  
   Example:
   C:\stable-diffusion-webui-reForge\models\LLM\llama-joycaption-beta-one
3. Apply settings → restart WebUI

> These paths are stored in your WebUI config — not inside this repo.

---

## 💾 Minimum System Requirements

### VRAM (JoyCaption)
| Mode | Typical VRAM |
|---|---:|
| 4-bit (Fastest) | ~10–12 GB |
| 8-bit (Balanced) | ~17–18 GB |
| Full FP16 | ~24–25 GB |

**Minimum recommended GPU:** **11 GB VRAM** (with Low VRAM mode)  
**Recommended:** **16 GB+ VRAM** for smoother multi-model workflows

### What “Low VRAM Mode” does
- reduces cached memory usage
- avoids holding multiple large components
- unloads more aggressively after use

---

## 🚀 Installation

### Install via WebUI
1. Open **Stable Diffusion WebUI ReForge**
2. Go to **Extensions → Install from URL**
3. Paste:
https://github.com/TdogCreations/ReForge-Prompt-Director
4. Click **Install**
5. Restart WebUI

---

## 🧰 Usage

### JoyCaption Ultra
Use it when you want **image → prompt** captions, or rewrite/constraint control.
- Enable JoyCaption in the JoyCaption UI
- Choose quantization (4-bit is fastest)
- Choose prompt style (SD Prompt / tag lists)
- Optional rewrite rules, required/banned constraints, and post-switches

### WD14 Tagger
Use it when you want **image → tags** injection.
- Choose image source:
- single ref image
- folder batch
- Pixiv cached images
- Choose prompt injection mode (append / prepend / replace)
- Optional: **Fusion Mode**
- Combine: add Image 2 tags
- Replace: replace character traits using Image 2

### QuickShot Prompt Director
Use it for consistent sets.
- age group strength
- time of day weighting
- camera / viewpoint controls
- lighting direction weighting
- blur steering

QuickShot works even if WD14/JoyCaption are disabled.

---

## 🧪 Batch & Prompt Matrix Support
Fully compatible with:
- ReForge batch execution
- prompt matrix
- dynamic prompts
- multi-image queues

Implementation detail:
- batch state is stored on the **Script instance**, not per-prompt,
to survive ReForge’s prompt/batch recreation behavior.

---

## 🔐 Privacy & Safety
- No telemetry
- No analytics
- No uploads
- Pixiv PHPSESSID is stored only in **WebUI settings**
- No credentials are written into the repo

---

## ⚠️ Common Mistakes / Fixes
- **WD14 does nothing** → ONNX/CSV paths not set in Settings
- **JoyCaption doesn’t load** → model folder path is wrong or missing files
- **Out of memory** → enable Low VRAM / use 4-bit / reduce batch size
- **Captions repeat** → restart WebUI after installing models (first-time load)

---

