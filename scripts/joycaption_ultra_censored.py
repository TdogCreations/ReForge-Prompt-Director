import gradio as gr
import torch, gc, os, hashlib, sys, re, json
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from modules import scripts, shared, script_callbacks

# ==========================================================
# JoyCaption Ultra v1.0.0 (Integrated) — FIXED FOR REFORGE BATCH
# Creator: TdogCreations
#
# Key Fixes in THIS build:
# ✅ Works even if ReForge/DynamicPrompts recreates `p` per batch (cfg stored on self)
# ✅ Base index resolved by searching p.all_prompts sequence (doesn't rely on batch_number)
# ✅ WD14 bridge supports PIL / np / {"name":path} / path string
# ✅ DEBUG prints so you can see if JoyCaption is actually running
# ==========================================================

# transformers import (keep script loadable even if HF/bnb is broken)
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
except Exception as e:
    AutoProcessor = None
    LlavaForConditionalGeneration = None
    BitsAndBytesConfig = None
    print(f"❌ [JoyCaption] transformers import failed -> script UI may load but captioning will not work: {e}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

SETTINGS_FILE = os.path.join(SCRIPT_DIR, "joycaption_ultra_settings.json")

try:
    from wd_quickshot_helper import apply_quickshot, AGE_PROMPT_MAP
except Exception:
    apply_quickshot = None
    AGE_PROMPT_MAP = {}

# ----------------------------------------------------------
# Optional WD14 inference (if user has ONNX + CSV set in opts)
# ----------------------------------------------------------
WD14_CHAR_DESIGN_KEYWORDS = [
    "hair", "eyes", "skin", "body", "dress", "shirt", "pants", "skirt",
    "shoes", "socks", "gloves", "hat", "tail", "ears", "wings", "horns",
    "jewelry", "glasses", "suit", "uniform", "armor", "bikini",
    "bra", "panties", "lingerie", "stockings", "collar", "choker",
    "coat", "jacket", "hoodie", "sweater", "kimono", "yukata",
    "apron", "cape", "scarf", "tie", "ribbon", "belt", "boots",
]

WD14_SEXUAL_BAN_WORDS = [
    "nsfw", "nude", "naked", "sex", "penis", "pussy", "vagina", "testicles",
    "cum", "ejaculation", "semen", "sperm", "intercourse", "anal", "vaginal",
    "blowjob", "handjob", "fellatio", "rape", "molestation", "loli", "shota", 
]

# ----------------------------------------------------------
# A1111 settings
# ----------------------------------------------------------
def on_ui_settings():
    section = ("joycaption", "JoyCaption")
    shared.opts.add_option(
        "joycaption_model_path",
        shared.OptionInfo(
            r"E:\Comfy Ui\ComfyUI_windows_portable\ComfyUI\models\LLM\llama-joycaption-beta-one",
            "JoyCaption Model Path",
            gr.Textbox,
            {"interactive": True},
            section=section,
        ),
    )

script_callbacks.on_ui_settings(on_ui_settings)


def _ensure_pil(img):
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    # gradio Image can be dict with {"name": "..."} sometimes
    if isinstance(img, dict):
        path = img.get("name") or img.get("path") or img.get("filename")
        if isinstance(path, str) and os.path.exists(path):
            try:
                with Image.open(path) as im:
                    return im.convert("RGB")
            except Exception:
                return None
    # if it's a filepath string
    if isinstance(img, str) and os.path.exists(img):
        try:
            with Image.open(img) as im:
                return im.convert("RGB")
        except Exception:
            return None
    # numpy array
    try:
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
    except Exception:
        pass
    return None


def _hash_image(img_pil: Image.Image) -> str:
    try:
        arr = np.array(img_pil.convert("RGB"))
        return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        return str(id(img_pil))


def _split_csv_line(line: str):
    parts, cur, in_q = [], "", False
    for ch in line:
        if ch == '"':
            in_q = not in_q
            continue
        if ch == "," and not in_q:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts]


def _is_minor_age_label(age_label: str) -> bool:
    if not age_label:
        return False
    s = str(age_label).lower()
    minor_markers = [
        "newborn", "baby", "young child", "child", "pre-teen", "preteen", "teen",
        "toddler", "kid", "loli", "shota",
    ]
    return any(k in s for k in minor_markers)


def _looks_like_bnb_dtype_crash(msg: str) -> bool:
    if not msg:
        return False
    m = str(msg)
    return (("same dtype" in m or "mat2" in m) and ("Half" in m) and ("Byte" in m or "Char" in m))


def _load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception as e:
        print(f"⚠️ [JoyCaption] Failed to load settings: {e}")
    return {}


def _save_settings(d: dict) -> bool:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"⚠️ [JoyCaption] Failed to save settings: {e}")
        return False


class JoyCaptionUltra(scripts.Script):
    def __init__(self):
        super().__init__()

        self.model = None
        self.processor = None
        self.current_path = None
        self.current_quant = None

        self._cap_cache = {}
        self._cap_cache_order = []
        self._cap_cache_max = 24

        self.wd_model = None
        self.wd_tags = None
        self.wd_current_onnx = None
        self.wd_current_csv = None

        # 🔥 REQUIRED: survive ReForge batch recreation
        self._active_cfg = None
        self._search_pos = 0
        self._global_cursor = 0
        self._job_sig = None

        # cached WD14 attribute discovery
        self._wd14_slot_attr_candidates = None
        self._wd14_paths_attr_candidates = None


    # ======================================================
    # Unload / VRAM helpers
    # ======================================================
    def _hard_unload(self):
        self.model = None
        self.processor = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _ui_unload_now(self):
        try:
            self._cap_cache.clear()
            self._cap_cache_order.clear()
        except Exception:
            pass
        self._hard_unload()
        return "✅ JoyCaption unloaded (model cleared + CUDA cache flushed)."

    def _ui_on_enabled_change(self, enabled_val):
        if not bool(enabled_val):
            self._active_cfg = None
            return self._ui_unload_now()
        return "✅ JoyCaption enabled (model loads on first use)."

    def _get_llm_device(self):
        try:
            if hasattr(self.model, "model") and self.model.model is not None:
                return next(self.model.model.parameters()).device
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cuda:0")

    def _force_mm_fp16_same_device(self):
        try:
            dev = self._get_llm_device()
            for attr in (
                "vision_tower", "vision_model", "visual", "siglip",
                "multi_modal_projector", "mm_projector", "projector"
            ):
                mt = getattr(self.model, attr, None)
                if mt is not None:
                    try:
                        mt.to(device=dev, dtype=torch.float16)
                    except Exception:
                        pass
        except Exception:
            pass

    # ======================================================
    # Model loading (skip vision/projector modules)
    # ======================================================
    def load_model(self, model_path: str, quantization: str, low_vram: bool):
        if AutoProcessor is None or LlavaForConditionalGeneration is None:
            print("❌ [JoyCaption] transformers not available -> cannot load JoyCaption model.")
            return False

        if self.model is not None and self.current_path == model_path and self.current_quant == quantization:
            return True

        self._hard_unload()

        skip_modules = [
            "vision_tower", "vision_model", "visual", "siglip",
            "multi_modal_projector", "mm_projector", "projector",
        ]

        try:
            device_map = "auto"
            max_memory = None
            if low_vram:
                max_memory = {0: "11GB", "cpu": "64GB"}

            if quantization == "4-bit (Fastest)":
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                if hasattr(bnb, "llm_int8_skip_modules"):
                    bnb.llm_int8_skip_modules = list(skip_modules)
                if hasattr(bnb, "bnb_4bit_skip_modules"):
                    bnb.bnb_4bit_skip_modules = list(skip_modules)

                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=bnb,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )

            elif quantization == "8-bit (Balanced)":
                bnb = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                if hasattr(bnb, "llm_int8_skip_modules"):
                    bnb.llm_int8_skip_modules = list(skip_modules)

                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=bnb,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )

            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            self.current_path = model_path
            self.current_quant = quantization

            self._force_mm_fp16_same_device()
            return True

        except Exception as e:
            print(f"❌ [JoyCaption] Model load failed: {e}")
            self._hard_unload()
            return False

    # ======================================================
    # WD14 (optional)
    # ======================================================
    def _load_wd14(self, onnx_path: str, csv_path: str):
        if (self.wd_model is not None) and (self.wd_current_onnx == onnx_path) and (self.wd_current_csv == csv_path):
            return True

        self.wd_model = None
        self.wd_tags = None
        self.wd_current_onnx = None
        self.wd_current_csv = None

        if not onnx_path or not os.path.exists(onnx_path):
            return False
        if not csv_path or not os.path.exists(csv_path):
            return False

        try:
            self.wd_model = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception:
            try:
                self.wd_model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            except Exception as e:
                print(f"⚠️ [JoyCaption] WD14 model load failed: {e}")
                return False

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            tags = []
            for li in lines[1:]:
                parts = _split_csv_line(li)
                if len(parts) >= 2:
                    tags.append(parts[1])
            self.wd_tags = tags
            self.wd_current_onnx = onnx_path
            self.wd_current_csv = csv_path
            return True
        except Exception as e:
            print(f"⚠️ [JoyCaption] WD14 tags CSV load failed: {e}")
            self.wd_model = None
            self.wd_tags = None
            return False

    def _wd14_predict(self, img_pil: Image.Image):
        if self.wd_model is None or self.wd_tags is None:
            return []
        try:
            img = np.array(img_pil.convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            inp_name = self.wd_model.get_inputs()[0].name
            pred = self.wd_model.run(None, {inp_name: img})[0][0]
            out = list(zip(self.wd_tags, pred.tolist()))
            out.sort(key=lambda x: x[1], reverse=True)
            return out
        except Exception as e:
            print(f"⚠️ [JoyCaption] WD14 predict failed: {e}")
            return []

    def _wd14_select_traits(self, preds, threshold: float, max_tags: int, scope: str):
        if not preds:
            return []
        chosen = []
        for tag, score in preds:
            if score < threshold:
                continue
            if scope == "Character Only":
                if not any(k in tag for k in WD14_CHAR_DESIGN_KEYWORDS):
                    continue
            chosen.append(tag.replace("_", " "))
            if len(chosen) >= max_tags:
                break
        return chosen

    # ======================================================
    # Caption cache + image token safety
    # ======================================================
    def _cap_cache_get(self, key: str):
        return self._cap_cache.get(key)

    def _cap_cache_put(self, key: str, caption: str):
        if key in self._cap_cache:
            return
        self._cap_cache[key] = caption
        self._cap_cache_order.append(key)
        if len(self._cap_cache_order) > self._cap_cache_max:
            old = self._cap_cache_order.pop(0)
            try:
                del self._cap_cache[old]
            except Exception:
                pass

    def _token_exists(self, token: str) -> bool:
        try:
            t = getattr(self.processor, "tokenizer", None)
            if t is None:
                return False
            tid = t.convert_tokens_to_ids(token)
            unk = getattr(t, "unk_token_id", None)
            if unk is None:
                return tid is not None and tid >= 0
            return tid is not None and tid != unk
        except Exception:
            return False

    def _pick_image_token(self) -> str:
        try:
            cfg_tok = getattr(self.model.config, "image_token", None)
            if isinstance(cfg_tok, str) and cfg_tok.strip() and self._token_exists(cfg_tok.strip()):
                return cfg_tok.strip()
        except Exception:
            pass

        try:
            t = getattr(self.processor, "tokenizer", None)
            if t is not None:
                add = list(getattr(t, "additional_special_tokens", []) or [])
                if "<image>" in add and self._token_exists("<image>"):
                    return "<image>"
                for s in add:
                    if isinstance(s, str) and "image" in s.lower() and self._token_exists(s):
                        return s
        except Exception:
            pass

        return "<image>"

    # ======================================================
    # Cleanup helpers
    # ======================================================
    def _cleanup_prompt_commas(self, text: str) -> str:
        if not text:
            return text
        t = text.strip()
        t = re.sub(r"\s*,\s*", ", ", t)
        t = re.sub(r"(,\s*){2,}", ", ", t)
        t = re.sub(r"\s{2,}", " ", t)
        return t.strip(" ,")

    def _apply_manual_replacements(self, text: str, rules_text: str, use_regex: bool = False) -> str:
        if not rules_text or not text:
            return text

        out = text
        for raw in rules_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "=>" in line:
                a, b = line.split("=>", 1)
            elif ":" in line:
                a, b = line.split(":", 1)
            else:
                continue

            a = a.strip()
            b = b.strip()
            if not a:
                continue

            if use_regex:
                try:
                    out = re.sub(a, b, out, flags=re.IGNORECASE)
                except Exception as e:
                    print(f"⚠️ [JoyCaption] Bad regex rule '{a}': {e}")
            else:
                out = out.replace(a, b)

        return out

    # ======================================================
    # HARD FORCE-GENDER FILTER (runs even if rewrite is OFF)
    # ======================================================
    def _enforce_force_gender_text(self, text: str, gender_mode: str) -> str:
        if not text:
            return text
        if gender_mode not in ("Force Male", "Force Female"):
            return text

        t = text

        if gender_mode == "Force Male":
            swaps = [
                (r"\b1girl\b", "1boy"),
                (r"\bgirl\b", "boy"),
                (r"\bwoman\b", "man"),
                (r"\bfemale\b", "male"),
                (r"\bshe\b", "he"),
                (r"\bher\b", "his"),
                (r"\bhers\b", "his"),
                (r"\bherself\b", "himself"),
            ]
            for a, b in swaps:
                t = re.sub(a, b, t, flags=re.IGNORECASE)

            strip_patterns = [
                r"\(?(very\s+)?long[_\s]hair(:[0-9.]+)?\)?",
                r"\(?(long[_\s]?bangs|bangs|side[_\s]?bangs)(:[0-9.]+)?\)?",
                r"\(?(ponytail|twin[_\s]?tails|twintails)(:[0-9.]+)?\)?",
                r"\(?(hair[_\s]ribbon|hair[_\s]bow)(:[0-9.]+)?\)?",
                r"\(?(breasts|boobs|bust)(:[0-9.]+)?\)?",
            ]
            for pat in strip_patterns:
                t = re.sub(pat, "", t, flags=re.IGNORECASE)

            if "short hair" not in t.lower() and "short_hair" not in t.lower():
                t = f"{t}, short hair"

        else:  # Force Female
            swaps = [
                (r"\b1boy\b", "1girl"),
                (r"\bboy\b", "girl"),
                (r"\bman\b", "woman"),
                (r"\bmale\b", "female"),
                (r"\bhe\b", "she"),
                (r"\bhis\b", "her"),
                (r"\bhim\b", "her"),
                (r"\bhimself\b", "herself"),
            ]
            for a, b in swaps:
                t = re.sub(a, b, t, flags=re.IGNORECASE)

        return self._cleanup_prompt_commas(t)

    # ======================================================
    # Captioning (with dtype fallback)
    # ======================================================
    def _caption_image(
        self,
        img_pil: Image.Image,
        model_path: str,
        quantization: str,
        prompt_style: str,
        max_len: int,
        low_vram: bool,
        force_rerun: bool,
        auto_fallback_fp16: bool,
        _already_retried: bool = False
    ):
        if img_pil is None:
            return ""

        key = f"{_hash_image(img_pil)}|{prompt_style}|{max_len}|{quantization}"
        if not force_rerun:
            cached = self._cap_cache_get(key)
            if cached:
                return cached

        if not self.load_model(model_path, quantization, low_vram):
            return ""

        image_token = self._pick_image_token()

        style_map = {
            "Descriptive": "Describe this image in great detail.",
            "Descriptive (Casual)": "Describe this image casually.",
            "Straightforward": "Give a straightforward description.",
            "Stable Diffusion Prompt": "Write a Stable Diffusion prompt as comma-separated tags/phrases.",
            "MidJourney": "Write a MidJourney style prompt.",
            "Danbooru tag list": "Output ONLY a comma-separated danbooru-style tag list.",
            "e621 tag list": "Output ONLY a comma-separated e621-style tag list.",
            "Rule34 tag list": "Output ONLY a comma-separated tag list.",
            "Booru-like tag list": "Output ONLY a comma-separated booru-style tag list.",
            "Art Critic": "Analyze this image as an art critic (short phrases).",
            "Product Listing": "Write a professional product listing description (short phrases).",
            "Social Media Post": "Write an engaging social media post (short phrases).",
        }
        user_msg = style_map.get(prompt_style, "Describe this image.")

        text_prompt = None
        try:
            tok = getattr(self.processor, "tokenizer", None)
            if tok is not None and hasattr(tok, "apply_chat_template"):
                messages = [{"role": "user", "content": f"{image_token}\n{user_msg}"}]
                text_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            text_prompt = None

        if not text_prompt:
            text_prompt = f"USER: {image_token}\n{user_msg}\nASSISTANT:"

        try:
            inputs = self.processor(text=text_prompt, images=img_pil.convert("RGB"), return_tensors="pt")

            dev = self._get_llm_device()
            for k in list(inputs.keys()):
                if isinstance(inputs[k], torch.Tensor):
                    if k == "pixel_values":
                        inputs[k] = inputs[k].contiguous().to(device=dev, dtype=torch.float16)
                    else:
                        inputs[k] = inputs[k].to(device=dev)

            eos_id = None
            try:
                eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            except Exception:
                eos_id = None

            with torch.inference_mode():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_len),
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=eos_id,
                )

            decoded = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            low = decoded.lower()
            cut = low.rfind("assistant")
            if cut != -1:
                decoded = decoded[cut + len("assistant"):].strip(" :\n\t")

            caption = decoded.strip()
            if caption:
                self._cap_cache_put(key, caption)
            return caption

        except Exception as e:
            msg = str(e)
            print(f"⚠️ [JoyCaption] Caption failed: {e}")

            if (
                auto_fallback_fp16
                and (not _already_retried)
                and quantization in ("4-bit (Fastest)", "8-bit (Balanced)")
                and _looks_like_bnb_dtype_crash(msg)
            ):
                print("⚠️ [JoyCaption] Quant dtype crash detected -> retrying once in Full 16-bit.")
                self._hard_unload()
                return self._caption_image(
                    img_pil=img_pil,
                    model_path=model_path,
                    quantization="None (Full 16-bit)",
                    prompt_style=prompt_style,
                    max_len=max_len,
                    low_vram=False,
                    force_rerun=True,
                    auto_fallback_fp16=False,
                    _already_retried=True
                )
            return ""

    # ======================================================
    # Rewrite caption (dtype fallback)
    # ======================================================
    def _rewrite_caption(
        self,
        caption: str,
        required_text: str,
        banned_text: str,
        hard_rules: str,
        max_new_tokens: int,
        model_path: str,
        quantization: str,
        auto_fallback_fp16: bool,
        low_vram: bool,
        _already_retried: bool = False
    ):
        if not caption:
            return caption
        if self.model is None or self.processor is None:
            return caption

        try:
            sys_prompt = (
                "Rewrite the caption to strictly follow the constraints.\n"
                "Output ONLY the final caption. Do not explain.\n"
            )

            user_parts = [f"Caption:\n{caption}\n"]
            if required_text:
                user_parts.append(f"REQUIRED:\n{required_text}\n")
            if banned_text:
                user_parts.append(f"BANNED:\n{banned_text}\n")
            if hard_rules:
                user_parts.append(f"RULES:\n{hard_rules}\n")

            full = sys_prompt + "\n" + ("\n".join(user_parts).strip())

            inputs = self.processor(text=full, return_tensors="pt")
            dev = self._get_llm_device()
            for k in list(inputs.keys()):
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(device=dev)

            with torch.inference_mode():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.92,
                )

            rewritten = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            rewritten = (
                rewritten.replace("{", "(").replace("}", ")")
                        .replace("[", "(").replace("]", ")")
                        .replace("|", " ")
            )
            return rewritten if rewritten else caption

        except Exception as e:
            msg = str(e)
            print(f"⚠️ [JoyCaption] Rewrite failed: {e}")

            if (
                auto_fallback_fp16
                and (not _already_retried)
                and quantization in ("4-bit (Fastest)", "8-bit (Balanced)")
                and _looks_like_bnb_dtype_crash(msg)
            ):
                print("⚠️ [JoyCaption] Rewrite dtype crash -> reloading Full 16-bit and retrying once.")
                self._hard_unload()
                if self.load_model(model_path, "None (Full 16-bit)", low_vram=False):
                    return self._rewrite_caption(
                        caption=caption,
                        required_text=required_text,
                        banned_text=banned_text,
                        hard_rules=hard_rules,
                        max_new_tokens=max_new_tokens,
                        model_path=model_path,
                        quantization="None (Full 16-bit)",
                        auto_fallback_fp16=False,
                        low_vram=False,
                        _already_retried=True
                    )
            return caption

    # ======================================================
    # Switch replacements (post)
    # ======================================================
    def _build_switch_replacements(self, gender_swap_mode: str, vaginal_to_oral: bool, cum_to_no_cum: bool, no_pubic_hair: bool) -> str:
        rules = []

        if gender_swap_mode == "Male → Female":
            rules += [
                r"\b1boy\b => 1girl",
                r"\bboy\b => girl",
                r"\bmale\b => female",
                r"\bman\b => woman",
                r"\bhe\b => she",
                r"\bhim\b => her",
                r"\bhimself\b => herself",
                r"\bhis\b => hers",
                r"\bson\b => daughter",
                r"\bfather\b => mother",
                r"\bbrother\b => sister",
                r"\buncle\b => aunt",
                r"\bgrandfather\b => grandmother",
                r"\bgrandpa\b => grandma",
                r"\bguy\b => gal",
                r"\bdude\b => chick",
                r"\bgentleman\b => lady",
                r"\bsir\b => ma'am",
                r"\bmister\b => miss",
                r"\bmr\b => ms",
                r"\bboys\b => girls",
                r"\bmales\b => females",
                r"\bmen\b => women",
                r"\bhe's\b => she's",
                r"\bhe'll\b => she'll",
            ]
        elif gender_swap_mode == "Female → Male":
            rules += [
                r"\b1girl\b => 1boy",
                r"\bgirl\b => boy",
                r"\bfemale\b => male",
                r"\bwoman\b => man",
                r"\bshe\b => he",
                r"\bher\b => his",
                r"\bhers\b => his",
                r"\bherself\b => himself",
                r"\bdaughter\b => son",
                r"\bmother\b => father",
                r"\bsister\b => brother",
                r"\baunt\b => uncle",
                r"\bgrandmother\b => grandfather",
                r"\bgrandma\b => grandpa",
                r"\bgal\b => guy",
                r"\bchick\b => dude",
                r"\blady\b => gentleman",
                r"\bma'am\b => sir",
                r"\bmiss\b => mister",
                r"\bms\b => mr",
                r"\bgirls\b => boys",
                r"\bfemales\b => males",
                r"\bwomen\b => men",
                r"\bshe's\b => he's",
                r"\bshe'll\b => he'll",
            ]

        if vaginal_to_oral:
            rules += [
                r"\bvaginal sex\b => oral sex",
                r"\bvaginal\b => oral",
                r"\bintercourse\b => oral sex",
                r"\bpenetration\b => oral sex",
            ]

        if cum_to_no_cum:
            rules += [
                r"\bcum\b => ",
                r"\bej(e)?aculation\b => ",
                r"\bsemen\b => ",
                r"\bsperm\b => ",
            ]

        if no_pubic_hair:
            rules += [
                r"\bpubic hair\b => no pubic hair, shaved pubic hair",
                r"\bpubes\b => no pubic hair, shaved pubic hair",
            ]

        return "\n".join(rules).strip()

    # ======================================================
    # Required / banned builder
    # ======================================================
    def _build_required_banned(self, **kwargs):
        rep_character = kwargs.get("rep_character", False)
        rep_setting = kwargs.get("rep_setting", False)
        rep_gender = kwargs.get("rep_gender", False)
        rep_age = kwargs.get("rep_age", False)
        rep_time = kwargs.get("rep_time", False)
        rep_view = kwargs.get("rep_view", False)

        gender_mode = kwargs.get("gender_mode", "Auto (from WD14)")

        age_group = kwargs.get("age_group", "None")
        remove_baby_props = kwargs.get("remove_baby_props", False)
        time_of_day = kwargs.get("time_of_day", "None")
        viewpoint = kwargs.get("viewpoint", "None")
        dutch_angle = kwargs.get("dutch_angle", False)

        use_wd14_traits = kwargs.get("use_wd14_traits", False)
        wd14_scope = kwargs.get("wd14_scope", "Character Only")
        wd14_threshold = kwargs.get("wd14_threshold", 0.35)
        wd14_max_tags = kwargs.get("wd14_max_tags", 12)

        character_override = kwargs.get("character_override", "")
        setting_override = kwargs.get("setting_override", "")
        extra_required = kwargs.get("extra_required", "")
        extra_banned = kwargs.get("extra_banned", "")

        img_pil = kwargs.get("img_pil", None)
        onnx_path = getattr(shared.opts, "wd_tagger_model_path", "")
        csv_path = getattr(shared.opts, "wd_tagger_csv_path", "")

        required, banned = [], []

        if use_wd14_traits and img_pil is not None and rep_character:
            if self._load_wd14(onnx_path, csv_path):
                preds = self._wd14_predict(img_pil)
                wd_traits = self._wd14_select_traits(preds, float(wd14_threshold), int(wd14_max_tags), str(wd14_scope))
                if wd_traits:
                    required += wd_traits

        if rep_character and character_override:
            required += [t.strip() for t in str(character_override).split(",") if t.strip()]

        if rep_setting and setting_override:
            required += [t.strip() for t in str(setting_override).split(",") if t.strip()]

        if rep_gender:
            if gender_mode == "Force Male":
                required.append("male")
                banned += [
                    "female", "girl", "1girl", "female body", "female anatomy",
                    "long hair", "very long hair", "long bangs", "bangs",
                    "ponytail", "twin tails", "twintails",
                    "hair ribbon", "hair bow",
                    "breasts", "boobs", "bust",
                ]
            elif gender_mode == "Force Female":
                required.append("female")
                banned += [
                    "male", "man", "boy", "1boy", "male body", "male anatomy",
                    "beard", "mustache", "facial hair",
                ]

        if rep_age and age_group and age_group != "None":
            required.append(str(age_group))
            if remove_baby_props:
                banned += ["pacifier", "bib"]

        if rep_time and time_of_day and time_of_day != "None":
            required.append(str(time_of_day))

        if rep_view and viewpoint and viewpoint != "None":
            required.append(str(viewpoint))
            if dutch_angle:
                required.append("dutch angle")

        if extra_required:
            required += [t.strip() for t in str(extra_required).split(",") if t.strip()]
        if extra_banned:
            banned += [t.strip() for t in str(extra_banned).split(",") if t.strip()]

        required_text = ", ".join(dict.fromkeys([r for r in required if r])).strip()
        banned_text = ", ".join(dict.fromkeys([b for b in banned if b])).strip()
        return required_text, banned_text

    # ======================================================
    # UI
    # ======================================================
    def ui(self, is_img2img):
        S = _load_settings()

        def _get(k, default):
            return S.get(k, default)

        def _safe_choice(val, choices, default):
            return val if val in choices else default

        quant_choices = ["4-bit (Fastest)", "8-bit (Balanced)", "None (Full 16-bit)"]
        style_choices = [
            "Descriptive",
            "Descriptive (Casual)",
            "Straightforward",
            "Stable Diffusion Prompt",
            "MidJourney",
            "Danbooru tag list",
            "e621 tag list",
            "Rule34 tag list",
            "Booru-like tag list",
            "Art Critic",
            "Product Listing",
            "Social Media Post",
        ]

        d_enabled = bool(_get("enabled", False))
        d_quant = _safe_choice(_get("quantization", "4-bit (Fastest)"), quant_choices, "4-bit (Fastest)")
        d_style = _safe_choice(_get("prompt_style", "Stable Diffusion Prompt"), style_choices, "Stable Diffusion Prompt")
        d_sync_wd14_image = bool(_get("sync_wd14_image", False))
        d_auto_fp16 = bool(_get("auto_fallback_fp16", True))
        d_max_len = int(_get("max_len", 128))
        d_mode = _safe_choice(_get("mode", "Append"), ["Append", "Prepend", "Replace"], "Append")

        with gr.Accordion("🛠️ JoyCaption", open=False):
            gr.HTML("""
<style>
#joy_unload_btn button {
  background: #c01818 !important;
  color: #fff !important;
  font-weight: 800 !important;
  border: 0 !important;
}
#joy_unload_btn button:hover { filter: brightness(1.08); }
#joy_save_btn button { font-weight: 800 !important; }
.joy_small_note { font-size: 12px; opacity: 0.8; }
</style>
""")

            with gr.Tabs():
                with gr.TabItem("Joy Core"):

                    enabled = gr.Checkbox(label="Enable JoyCaption", value=d_enabled)

                    with gr.Row():
                        quantization = gr.Dropdown(quant_choices, value=d_quant, label="Quantization")
                        prompt_style = gr.Dropdown(style_choices, value=d_style, label="Prompt Style")

                    sync_wd14_image = gr.Checkbox(
                        label="Use WD14 Image Slot as Joy Source (single image)",
                        value=d_sync_wd14_image
                    )

                    auto_fallback_fp16 = gr.Checkbox(
                        label="Auto-fallback to Full 16-bit if 4/8-bit crashes (recommended)",
                        value=d_auto_fp16
                    )

                    input_image = gr.Image(
                        label="Joy Source Image (txt2img only)",
                        type="pil",
                        visible=not is_img2img,
                    )

                    with gr.Row():
                        max_len = gr.Slider(32, 512, step=16, value=d_max_len, label="Max Caption Length")
                        mode = gr.Radio(["Append", "Prepend", "Replace"], value=d_mode, label="Injection Mode")

                with gr.TabItem("Controls"):
                    with gr.Tabs():
                        with gr.TabItem("Review / Rewrite"):
                            gr.Markdown(
                                "### Force-correct toggles (optional)\n"
                                "Turn these ON only when the model keeps getting something wrong.\n"
                                "<div class='joy_small_note'>Leaving them OFF keeps captions more natural and avoids unnecessary overrides.</div>"
                            )

                            preview_only = gr.Checkbox(label="Preview Only (print to console, don't inject)", value=bool(_get("preview_only", False)))

                            with gr.Row():
                                rewrite_caption = gr.Checkbox(label="Rewrite caption to match constraints", value=bool(_get("rewrite_caption", False)))
                                append_constraints = gr.Checkbox(label="Also append constraints blocks to final prompt", value=bool(_get("append_constraints", False)))

                            with gr.Row():
                                per_prompt_caption = gr.Checkbox(label="Caption each prompt separately (SLOW)", value=bool(_get("per_prompt_caption", False)))
                                use_wd14_batch_image = gr.Checkbox(label="Use WD14 batch image paths (if available)", value=bool(_get("use_wd14_batch_image", True)))

                            gr.Markdown("#### What to force-correct")
                            with gr.Row():
                                rep_character = gr.Checkbox(label="Force Character Traits", value=bool(_get("rep_character", False)))
                                rep_setting = gr.Checkbox(label="Force Setting / Location", value=bool(_get("rep_setting", False)))
                                rep_gender = gr.Checkbox(label="Force Gender", value=bool(_get("rep_gender", False)))
                                rep_age = gr.Checkbox(label="Force Age Group", value=bool(_get("rep_age", False)))
                                rep_time = gr.Checkbox(label="Force Time of Day", value=bool(_get("rep_time", False)))
                                rep_view = gr.Checkbox(label="Force Viewpoint / Angle", value=bool(_get("rep_view", False)))

                            with gr.Row():
                                gender_mode = gr.Dropdown(
                                    ["Auto (from WD14)", "Force Male", "Force Female", "None"],
                                    value=_safe_choice(_get("gender_mode", "Auto (from WD14)"),
                                                      ["Auto (from WD14)", "Force Male", "Force Female", "None"],
                                                      "Auto (from WD14)"),
                                    label="Gender Mode (used by rewrite + hard scrub)",
                                )
                                gender_swap = gr.Checkbox(label="Swap Gender (invert)", value=bool(_get("gender_swap", False)))

                            gr.Markdown("#### Manual overrides (optional)")
                            character_override = gr.Textbox(
                                label="Character Override (comma-separated traits)",
                                placeholder="e.g. blonde hair, blue eyes, ponytail, school uniform",
                                value=str(_get("character_override", "")),
                            )
                            setting_override = gr.Textbox(
                                label="Setting Override (scene / location)",
                                placeholder="e.g. classroom, city street, bedroom, beach",
                                value=str(_get("setting_override", "")),
                            )

                            extra_required = gr.Textbox(
                                label="Extra REQUIRED (comma-separated)",
                                placeholder="e.g. rainy night, neon lights",
                                value=str(_get("extra_required", "")),
                            )
                            extra_banned = gr.Textbox(
                                label="Extra BANNED (comma-separated)",
                                placeholder="e.g. watermark, text, logo",
                                value=str(_get("extra_banned", "")),
                            )

                            gr.Markdown("#### Manual Caption Override (optional)")
                            use_manual_caption = gr.Checkbox(label="Use Manual Caption (skip JoyCaption captioning)", value=bool(_get("use_manual_caption", False)))
                            manual_caption = gr.Textbox(
                                label="Manual Caption",
                                lines=3,
                                placeholder="This will be used instead of the model caption.",
                                value=str(_get("manual_caption", "")),
                            )

                        with gr.TabItem("QuickShot"):
                            gr.Markdown(
                                "### QuickShot\n"
                                "Adds gentle guidance for **age/time/viewpoint**.\n"
                                "<div class='joy_small_note'>Tip: strengths 0–3 are usually enough.</div>"
                            )

                            qs_nai_mode = gr.Checkbox(label="QuickShot NovelAI weighting mode", value=bool(_get("qs_nai_mode", False)))

                            age_choices = ["None"] + list(AGE_PROMPT_MAP.keys()) if AGE_PROMPT_MAP else ["None", "Young Adult", "Adult", "Older Adult"]
                            with gr.Row():
                                age_group = gr.Dropdown(age_choices, value=_safe_choice(_get("age_group", "None"), age_choices, "None"), label="Age Group")
                                age_strength = gr.Slider(0, 10, step=1, value=int(_get("age_strength", 0)), label="Age Strength (0–10)")
                                remove_baby_props = gr.Checkbox(label="🚫 Suppress Pacifier / Bib", value=bool(_get("remove_baby_props", False)))

                            with gr.Row():
                                time_of_day = gr.Dropdown(
                                    ["None", "Random", "dawn", "midday", "sunset", "midnight"],
                                    value=_safe_choice(_get("time_of_day", "None"), ["None", "Random", "dawn", "midday", "sunset", "midnight"], "None"),
                                    label="Time of Day",
                                )
                                time_weight = gr.Slider(0.0, 5.0, step=0.1, value=float(_get("time_weight", 1.0)), label="Time Weight")

                            with gr.Row():
                                viewpoint = gr.Dropdown(
                                    ["None", "Random", "from above", "from below", "from side", "from behind", "from the front"],
                                    value=_safe_choice(_get("viewpoint", "None"),
                                                      ["None", "Random", "from above", "from below", "from side", "from behind", "from the front"],
                                                      "None"),
                                    label="Viewpoint Angle",
                                )
                                viewpoint_scale = gr.Slider(0, 5, step=1, value=int(_get("viewpoint_scale", 0)), label="Viewpoint Scale (0–5)")
                                dutch_angle = gr.Checkbox(label="Dutch Angle", value=bool(_get("dutch_angle", False)))

                        with gr.TabItem("WD14"):
                            gr.Markdown("### WD14 help (optional)\nUsed mainly for trait consistency when rewriting.")

                            wd14_fallback_image = gr.Image(
                                label="WD14 Image Slot (also used as Joy source if toggle ON)",
                                type="pil",
                            )

                            use_wd14_traits = gr.Checkbox(label="Use WD14 traits to guide rewrite", value=bool(_get("use_wd14_traits", False)))

                            with gr.Row():
                                wd14_scope = gr.Dropdown(
                                    ["Character Only", "Character + Outfit", "Everything"],
                                    value=_safe_choice(_get("wd14_scope", "Character Only"),
                                                      ["Character Only", "Character + Outfit", "Everything"],
                                                      "Character Only"),
                                    label="WD14 Scope",
                                )
                                wd14_threshold = gr.Slider(0.05, 0.95, step=0.01, value=float(_get("wd14_threshold", 0.35)), label="WD14 Trait Threshold")
                                wd14_max_tags = gr.Slider(3, 30, step=1, value=int(_get("wd14_max_tags", 12)), label="Max WD14 Traits")

                        with gr.TabItem("Replacements"):
                            gr.Markdown("### Manual find/replace\nOne per line: `find => replace` (regex optional).")
                            with gr.Row():
                                apply_replace_before = gr.Checkbox(label="Apply manual replacements BEFORE rewrite", value=bool(_get("apply_replace_before", False)))
                                apply_replace_after = gr.Checkbox(label="Apply manual replacements AFTER rewrite", value=bool(_get("apply_replace_after", False)))
                                manual_replace_use_regex = gr.Checkbox(label="Use regex patterns", value=bool(_get("manual_replace_use_regex", False)))

                            manual_replacer = gr.Textbox(
                                label="Replacement rules",
                                lines=6,
                                placeholder="Example:\n1boy => 1girl\nmidnight => sunset\n\\bman\\b => woman",
                                value=str(_get("manual_replacer", "")),
                            )

                        with gr.TabItem("Switches"):
                            gr.Markdown("### One-click switches\nApplied after rewrite (if enabled).")
                            switch_gender_swap_mode = gr.Dropdown(
                                ["None", "Male → Female", "Female → Male"],
                                value=_safe_choice(_get("switch_gender_swap_mode", "None"),
                                                  ["None", "Male → Female", "Female → Male"],
                                                  "None"),
                                label="Gender term swap (post)",
                            )
                            switch_vaginal_to_oral = gr.Checkbox(label="Vaginal sex → Oral sex", value=bool(_get("switch_vaginal_to_oral", False)))
                            switch_cum_to_no_cum = gr.Checkbox(label="Cum → No cum", value=bool(_get("switch_cum_to_no_cum", False)))
                            switch_no_pubic_hair = gr.Checkbox(label="No pubic hair", value=bool(_get("switch_no_pubic_hair", False)))

                with gr.TabItem("VRAM"):
                    gr.Markdown("### VRAM / Cache")
                    joy_status = gr.Textbox(label="Status", value="Ready.", interactive=False)

                    unload_btn = gr.Button("🧹 Unload JoyCaption from VRAM NOW", elem_id="joy_unload_btn")

                    with gr.Row():
                        keep_vram = gr.Checkbox(label="Keep model in VRAM", value=bool(_get("keep_vram", False)))
                        force_rerun = gr.Checkbox(label="Force rerun (ignore cache)", value=bool(_get("force_rerun", False)))
                        low_vram = gr.Checkbox(label="Low VRAM mode", value=bool(_get("low_vram", False)))

                    save_btn = gr.Button("💾 Save JoyCaption Settings", elem_id="joy_save_btn")
                    save_status = gr.Textbox(label="Settings Save", value="", interactive=False)

            unload_btn.click(fn=self._ui_unload_now, inputs=[], outputs=[joy_status])
            enabled.change(fn=self._ui_on_enabled_change, inputs=[enabled], outputs=[joy_status])

            def _ui_save_settings(*vals):
                keys = [
                    "enabled","quantization","prompt_style","sync_wd14_image","auto_fallback_fp16","max_len","mode",
                    "preview_only","rewrite_caption","append_constraints","per_prompt_caption","use_wd14_batch_image",
                    "use_manual_caption","manual_caption",
                    "rep_character","rep_setting","rep_gender","rep_age","rep_time","rep_view",
                    "gender_mode","gender_swap",
                    "qs_nai_mode","age_group","age_strength","remove_baby_props",
                    "time_of_day","time_weight","viewpoint","viewpoint_scale","dutch_angle",
                    "use_wd14_traits","wd14_scope","wd14_threshold","wd14_max_tags",
                    "character_override","setting_override","extra_required","extra_banned",
                    "apply_replace_before","apply_replace_after","manual_replace_use_regex","manual_replacer",
                    "switch_gender_swap_mode","switch_vaginal_to_oral","switch_cum_to_no_cum","switch_no_pubic_hair",
                    "keep_vram","force_rerun","low_vram",
                ]
                d = {}
                for k, v in zip(keys, vals):
                    d[k] = v
                ok = _save_settings(d)
                return "Saved ✅" if ok else "Save failed ❌ (check console)"

            save_btn.click(
                fn=_ui_save_settings,
                inputs=[
                    enabled, quantization, prompt_style, sync_wd14_image, auto_fallback_fp16, max_len, mode,
                    preview_only, rewrite_caption, append_constraints, per_prompt_caption, use_wd14_batch_image,
                    use_manual_caption, manual_caption,
                    rep_character, rep_setting, rep_gender, rep_age, rep_time, rep_view,
                    gender_mode, gender_swap,
                    qs_nai_mode, age_group, age_strength, remove_baby_props,
                    time_of_day, time_weight, viewpoint, viewpoint_scale, dutch_angle,
                    use_wd14_traits, wd14_scope, wd14_threshold, wd14_max_tags,
                    character_override, setting_override, extra_required, extra_banned,
                    apply_replace_before, apply_replace_after, manual_replace_use_regex, manual_replacer,
                    switch_gender_swap_mode, switch_vaginal_to_oral, switch_cum_to_no_cum, switch_no_pubic_hair,
                    keep_vram, force_rerun, low_vram
                ],
                outputs=[save_status]
            )

        return [
            enabled, quantization, prompt_style, auto_fallback_fp16, input_image, sync_wd14_image, max_len, mode,
            preview_only, rewrite_caption, append_constraints,
            per_prompt_caption, use_wd14_batch_image,
            use_manual_caption, manual_caption,
            rep_character, rep_setting, rep_gender, rep_age, rep_time, rep_view,
            gender_mode, gender_swap,
            qs_nai_mode,
            age_group, age_strength, remove_baby_props,
            time_of_day, time_weight,
            viewpoint, viewpoint_scale, dutch_angle,
            wd14_fallback_image,
            use_wd14_traits, wd14_scope, wd14_threshold, wd14_max_tags,
            character_override, setting_override, extra_required, extra_banned,
            apply_replace_before, apply_replace_after, manual_replace_use_regex, manual_replacer,
            switch_gender_swap_mode, switch_vaginal_to_oral, switch_cum_to_no_cum, switch_no_pubic_hair,
            keep_vram, force_rerun, low_vram,
        ]

    # ======================================================
    # WD14 bridge (aggressive discovery)
    # ======================================================
    def _discover_wd14_candidates(self):
        if self._wd14_slot_attr_candidates is not None:
            return
        self._wd14_slot_attr_candidates = [
            "_wd14_slot_image_pil", "wd14_slot_image_pil", "_wd14_slot_image", "wd14_slot_image",
            "wd14_image", "wd14_current_image", "wd14_img", "wd14_slot",
            "wd14_slot_path", "wd14_image_path", "wd14_current_path"
        ]
        self._wd14_paths_attr_candidates = [
            "_wd14_batch_image_paths", "wd14_batch_image_paths",
            "wd14_paths", "wd14_image_paths", "wd14_batch_paths", "wd14_batch_files",
            "_batch_image_paths", "batch_image_paths",
        ]

    def _get_wd14_slot_image_from_anywhere(self, p):
        self._discover_wd14_candidates()

        # 1) attributes on p
        for name in self._wd14_slot_attr_candidates:
            try:
                v = getattr(p, name, None)
            except Exception:
                v = None
            im = _ensure_pil(v)
            if im is not None:
                return im
            # allow filepath string on p
            if isinstance(v, str) and os.path.exists(v):
                im = _ensure_pil(v)
                if im is not None:
                    return im

        # 2) modules.shared globals
        try:
            from modules import shared as _shared
            for name in self._wd14_slot_attr_candidates:
                v = getattr(_shared, name, None)
                im = _ensure_pil(v)
                if im is not None:
                    return im
                if isinstance(v, str) and os.path.exists(v):
                    im = _ensure_pil(v)
                    if im is not None:
                        return im
        except Exception:
            pass

        return None

    def _get_wd14_batch_paths_from_anywhere(self, p):
        self._discover_wd14_candidates()

        # 1) attributes on p
        for name in self._wd14_paths_attr_candidates:
            try:
                v = getattr(p, name, None)
            except Exception:
                v = None
            if v is not None:
                return v

        # 2) shared globals
        try:
            from modules import shared as _shared
            for name in self._wd14_paths_attr_candidates:
                v = getattr(_shared, name, None)
                if v is not None:
                    return v
        except Exception:
            pass

        return None

    def _get_batch_image_for_index(self, p, i: int):
        paths = self._get_wd14_batch_paths_from_anywhere(p)

        # dict mapping
        if isinstance(paths, dict):
            path = paths.get(i)
            if isinstance(path, dict):
                path = path.get("name") or path.get("path")
            if isinstance(path, str) and os.path.exists(path):
                return _ensure_pil(path)
            return None

        # list/tuple mapping
        if isinstance(paths, (list, tuple)):
            if 0 <= i < len(paths):
                path = paths[i]
                if isinstance(path, dict):
                    path = path.get("name") or path.get("path")
                if isinstance(path, str) and os.path.exists(path):
                    return _ensure_pil(path)
        return None

    # ======================================================
    # Resolve base_index robustly (no batch_number required)
    # ======================================================
    def _resolve_base_index(self, p, prompts):
        bsz = len(prompts)

        allp = getattr(p, "all_prompts", None)
        if isinstance(allp, list) and allp and isinstance(prompts, list) and bsz > 0:
            # search from last position to avoid matching first duplicate
            start = max(0, int(self._search_pos or 0))
            limit = len(allp) - bsz
            for i in range(start, limit + 1):
                if allp[i:i+bsz] == prompts:
                    self._search_pos = i + bsz
                    return i
            # fallback from 0 if not found
            for i in range(0, limit + 1):
                if allp[i:i+bsz] == prompts:
                    self._search_pos = i + bsz
                    return i

        # worst-case fallback
        bi = getattr(p, "batch_index", None)
        if isinstance(bi, int):
            return bi * bsz

        return 0

    # ======================================================
    # Main process (store cfg on self + p)
    # ======================================================
    def process(
        self,
        p,
        enabled, quantization, prompt_style, auto_fallback_fp16, input_image, sync_wd14_image, max_len, mode,
        preview_only, rewrite_caption, append_constraints,
        per_prompt_caption, use_wd14_batch_image,
        use_manual_caption, manual_caption,
        rep_character, rep_setting, rep_gender, rep_age, rep_time, rep_view,
        gender_mode, gender_swap,
        qs_nai_mode,
        age_group, age_strength, remove_baby_props,
        time_of_day, time_weight,
        viewpoint, viewpoint_scale, dutch_angle,
        wd14_fallback_image,
        use_wd14_traits, wd14_scope, wd14_threshold, wd14_max_tags,
        character_override, setting_override, extra_required, extra_banned,
        apply_replace_before, apply_replace_after, manual_replace_use_regex, manual_replacer,
        switch_gender_swap_mode, switch_vaginal_to_oral, switch_cum_to_no_cum, switch_no_pubic_hair,
        keep_vram, force_rerun, low_vram,
    ):
        if not enabled:
            self._active_cfg = None
            return

        model_path = getattr(shared.opts, "joycaption_model_path", None)
        if not model_path or not os.path.exists(model_path):
            print("⚠️ [JoyCaption] Invalid model path.")
            return

        extra_instr = ""
        if apply_quickshot is not None:
            try:
                extra_instr = apply_quickshot(
                    age_group=str(age_group),
                    age_strength=int(age_strength),
                    time_of_day=str(time_of_day),
                    time_weight=float(time_weight),
                    viewpoint=str(viewpoint),
                    viewpoint_scale=int(viewpoint_scale),
                    dutch_angle=bool(dutch_angle),
                    nai_mode=bool(qs_nai_mode),
                    remove_baby_props=bool(remove_baby_props),
                )
            except Exception:
                extra_instr = ""

        cfg = {
            "enabled": True,
            "model_path": model_path,
            "quantization": quantization,
            "prompt_style": prompt_style,
            "auto_fallback_fp16": bool(auto_fallback_fp16),
            "input_image": input_image,
            "sync_wd14_image": bool(sync_wd14_image),
            "max_len": int(max_len),
            "mode": mode,

            "preview_only": bool(preview_only),
            "rewrite_caption": bool(rewrite_caption),
            "append_constraints": bool(append_constraints),
            "per_prompt_caption": bool(per_prompt_caption),
            "use_wd14_batch_image": bool(use_wd14_batch_image),

            "use_manual_caption": bool(use_manual_caption),
            "manual_caption": str(manual_caption or "").strip(),

            "rep_character": bool(rep_character),
            "rep_setting": bool(rep_setting),
            "rep_gender": bool(rep_gender),
            "rep_age": bool(rep_age),
            "rep_time": bool(rep_time),
            "rep_view": bool(rep_view),

            "gender_mode": str(gender_mode),
            "gender_swap": bool(gender_swap),

            "age_group": str(age_group),
            "remove_baby_props": bool(remove_baby_props),
            "time_of_day": str(time_of_day),
            "viewpoint": str(viewpoint),
            "dutch_angle": bool(dutch_angle),

            "wd14_fallback_image": wd14_fallback_image,
            "use_wd14_traits": bool(use_wd14_traits),
            "wd14_scope": str(wd14_scope),
            "wd14_threshold": float(wd14_threshold),
            "wd14_max_tags": int(wd14_max_tags),

            "character_override": str(character_override or ""),
            "setting_override": str(setting_override or ""),
            "extra_required": str(extra_required or ""),
            "extra_banned": str(extra_banned or ""),

            "apply_replace_before": bool(apply_replace_before),
            "apply_replace_after": bool(apply_replace_after),
            "manual_replace_use_regex": bool(manual_replace_use_regex),
            "manual_replacer": str(manual_replacer or ""),

            "switch_gender_swap_mode": str(switch_gender_swap_mode),
            "switch_vaginal_to_oral": bool(switch_vaginal_to_oral),
            "switch_cum_to_no_cum": bool(switch_cum_to_no_cum),
            "switch_no_pubic_hair": bool(switch_no_pubic_hair),

            "extra_instr": str(extra_instr or ""),

            "keep_vram": bool(keep_vram),
            "force_rerun": bool(force_rerun),
            "low_vram": bool(low_vram),
        }

        # store on self (SURVIVES new `p`)
        self._active_cfg = cfg
        self._search_pos = 0

        # also store on this p for normal pipelines
        try:
            p._joycap_batch_cfg = cfg
        except Exception:
            pass

        print("✅ [JoyCaption] process(): cfg stored (self + p).")

    # ======================================================
    # process_batch (caption + inject per batch)
    # ======================================================
    def process_batch(self, p, *args, **kwargs):
        cfg = getattr(p, "_joycap_batch_cfg", None)
        if not isinstance(cfg, dict):
            cfg = self._active_cfg
        if not isinstance(cfg, dict) or not cfg.get("enabled"):
            return

        # find prompts list in kwargs/args
        prompts = kwargs.get("prompts", None)
        if prompts is None:
            for a in args:
                if isinstance(a, (list, tuple)) and len(a) and all(isinstance(x, str) for x in a):
                    prompts = list(a)
                    break
        if not isinstance(prompts, list) or len(prompts) == 0:
            return

        batch_size = len(prompts)
        base_index = self._resolve_base_index(p, prompts)

        print(f"🧩 [JoyCaption] process_batch(): batch_size={batch_size} base_index={base_index}")

        # unpack cfg
        model_path = cfg["model_path"]
        quantization = cfg["quantization"]
        prompt_style = cfg["prompt_style"]
        auto_fallback_fp16 = bool(cfg["auto_fallback_fp16"])
        max_len = int(cfg["max_len"])
        mode = cfg["mode"]

        preview_only = bool(cfg["preview_only"])
        rewrite_caption = bool(cfg["rewrite_caption"])
        append_constraints = bool(cfg["append_constraints"])
        per_prompt_caption = bool(cfg["per_prompt_caption"])
        use_wd14_batch_image = bool(cfg["use_wd14_batch_image"])

        use_manual_caption = bool(cfg["use_manual_caption"])
        manual_caption = str(cfg["manual_caption"] or "").strip()

        rep_character = bool(cfg["rep_character"])
        rep_setting = bool(cfg["rep_setting"])
        rep_gender = bool(cfg["rep_gender"])
        rep_age = bool(cfg["rep_age"])
        rep_time = bool(cfg["rep_time"])
        rep_view = bool(cfg["rep_view"])

        gender_mode = str(cfg["gender_mode"])
        age_group = str(cfg["age_group"])
        remove_baby_props = bool(cfg["remove_baby_props"])
        time_of_day = str(cfg["time_of_day"])
        viewpoint = str(cfg["viewpoint"])
        dutch_angle = bool(cfg["dutch_angle"])

        wd14_fallback_image = cfg.get("wd14_fallback_image", None)
        sync_wd14_image = bool(cfg.get("sync_wd14_image", False))
        input_image = cfg.get("input_image", None)

        use_wd14_traits = bool(cfg["use_wd14_traits"])
        wd14_scope = str(cfg["wd14_scope"])
        wd14_threshold = float(cfg["wd14_threshold"])
        wd14_max_tags = int(cfg["wd14_max_tags"])

        character_override = str(cfg["character_override"] or "")
        setting_override = str(cfg["setting_override"] or "")
        extra_required = str(cfg["extra_required"] or "")
        extra_banned = str(cfg["extra_banned"] or "")

        apply_replace_before = bool(cfg["apply_replace_before"])
        apply_replace_after = bool(cfg["apply_replace_after"])
        manual_replace_use_regex = bool(cfg["manual_replace_use_regex"])
        manual_replacer = str(cfg["manual_replacer"] or "")

        switch_gender_swap_mode = str(cfg["switch_gender_swap_mode"])
        switch_vaginal_to_oral = bool(cfg["switch_vaginal_to_oral"])
        switch_cum_to_no_cum = bool(cfg["switch_cum_to_no_cum"])
        switch_no_pubic_hair = bool(cfg["switch_no_pubic_hair"])

        extra_instr = str(cfg.get("extra_instr", "") or "")

        keep_vram = bool(cfg["keep_vram"])
        force_rerun = bool(cfg["force_rerun"])
        low_vram = bool(cfg["low_vram"])

        if low_vram:
            max_len = min(int(max_len), 128)

        def get_img_for_global_index(gidx: int):
            # 1) WD14 batch path (per image)
            if use_wd14_batch_image:
                bi = self._get_batch_image_for_index(p, gidx)
                if bi is not None:
                    return bi

            # 2) WD14 slot image (current)
            if sync_wd14_image:
                wd_img = self._get_wd14_slot_image_from_anywhere(p)
                if wd_img is not None:
                    return wd_img
                if wd14_fallback_image is not None:
                    return _ensure_pil(wd14_fallback_image)

            # 3) txt2img Joy input image
            if input_image is not None:
                return _ensure_pil(input_image)

            # 4) img2img init image
            try:
                if hasattr(p, "init_images") and p.init_images:
                    return _ensure_pil(p.init_images[0])
            except Exception:
                pass

            # 5) fallback slot
            if wd14_fallback_image is not None:
                return _ensure_pil(wd14_fallback_image)

            return None

        captions = [""] * batch_size

        if use_manual_caption and manual_caption:
            captions = [manual_caption for _ in range(batch_size)]
        else:
            if per_prompt_caption:
                for bi in range(batch_size):
                    gidx = base_index + bi
                    img_pil = _ensure_pil(get_img_for_global_index(gidx))
                    if img_pil is None:
                        print(f"⚠️ [JoyCaption] No image found for gidx={gidx}")
                        captions[bi] = ""
                        continue
                    captions[bi] = self._caption_image(
                        img_pil=img_pil,
                        model_path=model_path,
                        quantization=quantization,
                        prompt_style=prompt_style,
                        max_len=int(max_len),
                        low_vram=bool(low_vram),
                        force_rerun=bool(force_rerun),
                        auto_fallback_fp16=bool(auto_fallback_fp16),
                    )
            else:
                gidx0 = base_index
                img0 = _ensure_pil(get_img_for_global_index(gidx0))
                if img0 is None:
                    print(f"⚠️ [JoyCaption] No image found for base gidx={gidx0}")
                    cap0 = ""
                else:
                    cap0 = self._caption_image(
                        img_pil=img0,
                        model_path=model_path,
                        quantization=quantization,
                        prompt_style=prompt_style,
                        max_len=int(max_len),
                        low_vram=bool(low_vram),
                        force_rerun=bool(force_rerun),
                        auto_fallback_fp16=bool(auto_fallback_fp16),
                    )
                captions = [cap0 for _ in range(batch_size)]

        if apply_replace_before and manual_replacer:
            captions = [
                self._cleanup_prompt_commas(
                    self._apply_manual_replacements(c, manual_replacer, use_regex=bool(manual_replace_use_regex))
                ) for c in captions
            ]

        if rep_gender and gender_mode in ("Force Male", "Force Female"):
            captions = [self._enforce_force_gender_text(c, gender_mode) for c in captions]

        if rewrite_caption:
            if not self.load_model(model_path, quantization, bool(low_vram)):
                return

            rewritten_caps = []
            for bi in range(batch_size):
                gidx = base_index + bi
                img_i = _ensure_pil(get_img_for_global_index(gidx))

                required_text, banned_text = self._build_required_banned(
                    rep_character=rep_character,
                    rep_setting=rep_setting,
                    rep_gender=rep_gender,
                    rep_age=rep_age,
                    rep_time=rep_time,
                    rep_view=rep_view,
                    gender_mode=gender_mode,
                    age_group=age_group,
                    remove_baby_props=remove_baby_props,
                    time_of_day=time_of_day,
                    viewpoint=viewpoint,
                    dutch_angle=dutch_angle,
                    use_wd14_traits=use_wd14_traits,
                    wd14_scope=wd14_scope,
                    wd14_threshold=wd14_threshold,
                    wd14_max_tags=wd14_max_tags,
                    character_override=character_override,
                    setting_override=setting_override,
                    extra_required=extra_required,
                    extra_banned=extra_banned,
                    img_pil=img_i,
                )

                if _is_minor_age_label(str(age_group)):
                    required_text = (required_text + ", " if required_text else "") + "fully clothed, non-sexual, safe"
                    banned_text = (banned_text + ", " if banned_text else "") + ", ".join(WD14_SEXUAL_BAN_WORDS)

                hard_rules = []
                if rep_gender and gender_mode == "Force Male":
                    hard_rules.append("The subject MUST be male (a man/boy).")
                if rep_gender and gender_mode == "Force Female":
                    hard_rules.append("The subject MUST be female (a girl/woman).")
                if rep_age and age_group and age_group != "None":
                    hard_rules.append(f"The age MUST match: {age_group}.")
                if rep_time and time_of_day and time_of_day != "None":
                    hard_rules.append(f"The time of day MUST match: {time_of_day}.")
                if rep_view and viewpoint and viewpoint != "None":
                    hard_rules.append(f"The camera viewpoint MUST match: {viewpoint}.")
                if rep_setting and setting_override:
                    hard_rules.append("The setting MUST match the provided Setting Override.")
                if rep_character and character_override:
                    hard_rules.append("Character appearance MUST match the provided Character Override.")
                if _is_minor_age_label(str(age_group)):
                    hard_rules.append("Keep the output fully non-sexual.")
                hard_rules_text = " ".join(hard_rules).strip()

                c = captions[bi] or ""
                if (required_text or banned_text or hard_rules_text) and c:
                    c = self._rewrite_caption(
                        caption=c,
                        required_text=required_text,
                        banned_text=banned_text,
                        hard_rules=hard_rules_text,
                        max_new_tokens=min(180, int(max_len)),
                        model_path=model_path,
                        quantization=quantization,
                        auto_fallback_fp16=bool(auto_fallback_fp16),
                        low_vram=bool(low_vram),
                    )
                rewritten_caps.append(c)

            captions = rewritten_caps

        # switches post rewrite (disabled for minors)
        if _is_minor_age_label(str(age_group)):
            switch_rules = self._build_switch_replacements(
                gender_swap_mode=switch_gender_swap_mode,
                vaginal_to_oral=False,
                cum_to_no_cum=False,
                no_pubic_hair=False,
            )
        else:
            switch_rules = self._build_switch_replacements(
                gender_swap_mode=switch_gender_swap_mode,
                vaginal_to_oral=switch_vaginal_to_oral,
                cum_to_no_cum=switch_cum_to_no_cum,
                no_pubic_hair=switch_no_pubic_hair,
            )

        if switch_rules:
            captions = [
                self._cleanup_prompt_commas(
                    self._apply_manual_replacements(c, switch_rules, use_regex=True)
                ) for c in captions
            ]
            if switch_cum_to_no_cum and (not _is_minor_age_label(str(age_group))):
                fixed = []
                for c in captions:
                    if c and ("no cum" not in c.lower()):
                        fixed.append(self._cleanup_prompt_commas(f"{c}, no cum"))
                    else:
                        fixed.append(c)
                captions = fixed

        if apply_replace_after and manual_replacer:
            captions = [
                self._cleanup_prompt_commas(
                    self._apply_manual_replacements(c, manual_replacer, use_regex=bool(manual_replace_use_regex))
                ) for c in captions
            ]

        if preview_only:
            print("\n" + "=" * 70)
            print(f"📝 [JoyCaption PREVIEW] base_index={base_index}")
            for bi, c in enumerate(captions):
                print(f"  [{base_index + bi}] {c}")
            if extra_instr:
                print("\n✨ EXTRA:", extra_instr)
            print("=" * 70 + "\n")
            return

        # inject into this batch list + p.all_prompts
        for bi in range(batch_size):
            base = prompts[bi] or ""
            cap = (captions[bi] or "").replace("|", " ").replace("||", " ").strip()

            if not cap:
                continue

            if append_constraints:
                gidx = base_index + bi
                img_i = _ensure_pil(get_img_for_global_index(gidx))
                required_text, banned_text = self._build_required_banned(
                    rep_character=rep_character,
                    rep_setting=rep_setting,
                    rep_gender=rep_gender,
                    rep_age=rep_age,
                    rep_time=rep_time,
                    rep_view=rep_view,
                    gender_mode=gender_mode,
                    age_group=age_group,
                    remove_baby_props=remove_baby_props,
                    time_of_day=time_of_day,
                    viewpoint=viewpoint,
                    dutch_angle=dutch_angle,
                    use_wd14_traits=use_wd14_traits,
                    wd14_scope=wd14_scope,
                    wd14_threshold=wd14_threshold,
                    wd14_max_tags=wd14_max_tags,
                    character_override=character_override,
                    setting_override=setting_override,
                    extra_required=extra_required,
                    extra_banned=extra_banned,
                    img_pil=img_i,
                )
                if _is_minor_age_label(str(age_group)):
                    required_text = (required_text + ", " if required_text else "") + "fully clothed, non-sexual, safe"
                    banned_text = (banned_text + ", " if banned_text else "") + ", ".join(WD14_SEXUAL_BAN_WORDS)

                blocks = []
                if required_text:
                    blocks.append(f"[REQUIRED: {required_text}]")
                if banned_text:
                    blocks.append(f"[BANNED: {banned_text}]")
                if extra_instr:
                    blocks.append(f"[EXTRA: {extra_instr}]")
                if blocks:
                    cap = f"{cap} " + " ".join(blocks)

            if mode == "Append":
                new_p = f"{base}, {cap}".strip(", ").strip() if base else cap
            elif mode == "Prepend":
                new_p = f"{cap}, {base}".strip(", ").strip() if base else cap
            else:
                new_p = cap

            if rep_gender and gender_mode in ("Force Male", "Force Female"):
                new_p = self._enforce_force_gender_text(new_p, gender_mode)

            prompts[bi] = new_p

            try:
                gi = base_index + bi
                if hasattr(p, "all_prompts") and isinstance(p.all_prompts, list):
                    if 0 <= gi < len(p.all_prompts):
                        p.all_prompts[gi] = new_p
                if hasattr(p, "prompts") and isinstance(p.prompts, list):
                    if 0 <= gi < len(p.prompts):
                        p.prompts[gi] = new_p
                if batch_size == 1 and hasattr(p, "prompt"):
                    p.prompt = new_p
            except Exception:
                pass

        print("✅ [JoyCaption] Injected prompts for this batch.")

    def postprocess(self, p, processed, *args):
        cfg = getattr(p, "_joycap_batch_cfg", None)
        if isinstance(cfg, dict):
            if (not cfg.get("keep_vram")) or cfg.get("low_vram"):
                self._hard_unload()
