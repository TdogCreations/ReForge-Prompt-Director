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

## 🚀 Installation

### Install via WebUI
1. Open **Stable Diffusion WebUI ReForge**
2. Go to **Extensions → Install from URL**
3. Paste:
https://github.com/TdogCreations/ReForge-Prompt-Director
4. Click **Install**
5. Restart WebUI
---

## 📥 Required Model Downloads

This extension does **not** bundle large models. You install them manually.

### 1) WD14 Tagger (Required for WD14)

#### ✅ Recommended (best quality)
**WD EVA02 Large Tagger v3**
- ONNX:  
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3
- CSV:  
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3

#### 🟡 Alternative (older / slightly lighter)
**WD EVA02 Large Tagger v2**
- ONNX:  
  https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3
- CSV:  
  https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2

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
https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava

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

## 🎨 Pixiv Downloader (Optional) — PHPSESSID Setup

WD14 Tagger can download images directly from a Pixiv artwork URL and use them as the batch image source.

### ✅ What you need
Pixiv requires a login session cookie called **PHPSESSID**.

This extension uses your PHPSESSID **only to fetch the images you already have access to**.

### 🔐 How to get your PHPSESSID (Chrome / Edge)
1. Log into Pixiv in your browser.
2. Open Pixiv and go to any page (ex: your bookmarks or an artwork page).
3. Press **F12** to open **Developer Tools**
4. Go to the **Application** tab  
   (in some browsers it may be under **Storage**)
5. In the left sidebar, open:
   - **Cookies**
   - Select: `https://www.pixiv.net`
6. Find the cookie named: **PHPSESSID**
7. Copy the **Value** (it will look like a long string such as `1234567_abcd...`)

### ⚙️ Where to paste it in WebUI
1. WebUI → **Settings → WD14 Tagger**
2. Paste into: **Pixiv PHPSESSID**
3. Apply settings → restart WebUI

✅ Your PHPSESSID is stored in your local WebUI settings (not this repo).  
⚠️ Treat it like a password — **never share it** and **don’t commit config files to GitHub**.

---

## 🚀 Pixiv → Prompt Workflow (Direct-to-Prompt)

### Goal
Paste a Pixiv artwork URL → download all pages → auto-inject tags (and optionally JoyCaption captions) into your prompts.

### Steps
1. Enable **WD14 Tagger**
2. Open **Batch Sources → Pixiv Gallery**
3. Turn ON **Enable Pixiv**
4. Paste a Pixiv artwork URL, e.g.:
   - `https://www.pixiv.net/en/artworks/123456789`
5. Click **Download Images**
6. Choose your indexing mode:
   - **Increment** = each prompt uses the next Pixiv page
   - **Random** = random page per prompt
   - **Fixed** = always use the same page
7. Generate normally (batch, matrix, dynamic prompts supported)

### Optional: Add JoyCaption on the same Pixiv images
To make JoyCaption analyze the *same image that WD14 selected*:
- Enable JoyCaption
- Enable **Use WD14 batch image paths (if available)**

Now you get:
- WD14 tags + JoyCaption captions both aligned per prompt index

## 🛡️ Pixiv Safety & Rate Limits (Read This)

This extension includes a **basic safety throttle** to reduce the chance of triggering Pixiv rate limits.

### ✅ Built-in safety feature
- **Safety Delay** slider adds a pause between image downloads.
- This helps avoid hammering Pixiv’s servers with rapid-fire requests.

### ⚠️ Important warning (protect your Pixiv account)
Pixiv can rate-limit or flag accounts that download too aggressively.

**Avoid:**
- Downloading *hundreds* of images in one go
- Repeated downloads of the same large gallery
- Running multiple download sessions at the same time

**Recommended:**
- Use **Safety Delay = 2–5 seconds**
- Download in smaller batches (ex: 10–30 images), then generate
- If downloads start failing, stop and wait before trying again

### 🔒 Privacy reminder
Your **PHPSESSID** is a login session cookie.
- Treat it like a password
- Never post it in screenshots
- Never commit WebUI config files to GitHub

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


