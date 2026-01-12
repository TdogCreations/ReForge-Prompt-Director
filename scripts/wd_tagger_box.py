import gradio as gr
import numpy as np
import cv2
import pandas as pd
import onnxruntime as ort
import os
import random
import time
import re
import requests
import json
import sys
from PIL import Image
from modules import scripts, shared, script_callbacks

SETTINGS_FILE = os.path.join(scripts.basedir(), "tagger_quickshot_settings.json")

print("🔥🔥🔥 WD14 SCRIPT IMPORTED 🔥🔥🔥")

# ==========================================================
# Add Settings entries so you can set .onnx and tags.csv
# ==========================================================
def on_ui_settings():
    section = ("wd_tagger", "WD14 Tagger")
    shared.opts.add_option(
        "wd_tagger_model_path",
        shared.OptionInfo("", "WD14 Model Path (.onnx)", section=section),
    )
    shared.opts.add_option(
        "wd_tagger_csv_path",
        shared.OptionInfo("", "WD14 Tags Path (.csv)", section=section),
    )
    # Optional Pixiv PHPSESSID (if you use Pixiv tab)
    shared.opts.add_option(
        "wd_tagger_pixiv_phpid",
        shared.OptionInfo("", "Pixiv PHPSESSID (for Pixiv download)", section=section),
    )

script_callbacks.on_ui_settings(on_ui_settings)

# ==========================================================
# Import helper from same folder
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from wd_quickshot_helper import apply_quickshot, AGE_PROMPT_MAP


class WDTaggerBox(scripts.Script):
    def __init__(self):
        self.model = None
        self.tags = None

        self.censor_keywords = ["censored", "bar censor", "mosaic censoring", "white censor", "blurry censor"]
        self.char_design_keywords = [
            "hair", "eyes", "skin", "body", "dress", "shirt", "pants", "skirt", "shoes", "socks", "gloves",
            "hat", "tail", "ears", "wings", "horns", "jewelry", "glasses", "suit", "uniform", "armor",
            "bikini", "lingerie", "cleavage", "breasts", "thighs", "leg", "arm", "face", "makeup", "ribbon"
        ]
        self.pubic_hair_tags = [
            "pubic hair", "male pubic hair", "female pubic hair", "armpit hair", "underarm hair", "hairy",
            "hairy male", "hairy female", "hair on chest", "chest hair", "pubic_hair", "pubic", "pubic_area"
        ]

        self.cached_target_files = []
        self.batch_offset = 0
        self._counter_signature = None

        # ✅ ReForge batch-safe state
        self._active_cfg = None
        self._search_pos = 0
        self._global_cursor = 0
        self._job_sig = None

    def title(self):
        return "WD14 Tagger (Integrated) v2.0.7 - ReForge Batch Fix + JoyCaption Bridge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # --------------------------
    # Counter helpers
    # --------------------------
    def _files_signature(self, files, pixiv_enabled, pixiv_url, batch_enabled, folder_path, sort_by, sort_dir):
        first = files[0] if files else ""
        last = files[-1] if files else ""
        return (
            len(files), first, last,
            bool(pixiv_enabled), str(pixiv_url or ""),
            bool(batch_enabled), str(folder_path or ""),
            str(sort_by or ""), str(sort_dir or "")
        )

    def reset_counter(self):
        self.batch_offset = 0
        self._counter_signature = None
        return 0

    # --------------------------
    # Sorting helper
    # --------------------------
    def sort_files(self, files, sort_by, sort_dir):
        if not files:
            return files
        reverse = (sort_dir == "Descending")
        try:
            if sort_by == "Name":
                return sorted(files, key=lambda p: os.path.basename(p).lower(), reverse=reverse)
            elif sort_by == "Date Modified":
                return sorted(files, key=lambda p: os.path.getmtime(p), reverse=reverse)
            elif sort_by == "File Size":
                return sorted(files, key=lambda p: os.path.getsize(p), reverse=reverse)
        except Exception:
            pass
        return files

    # ======================================================
    # UI
    # ======================================================
    def ui(self, is_img2img):
        s = self.load_settings()

        with gr.Accordion("🏷️ WD14 Tagger Auto-Inject", open=False):
            with gr.Row():
                enabled = gr.Checkbox(label="Enable Tagger", value=s.get("enabled", False))
                allow_nsfw = gr.Checkbox(label="Allow NSFW Tags", value=s.get("allow_nsfw", True))
                char_only = gr.Checkbox(label="👤 Character Design Only", value=s.get("char_only", False))
                decensor_mode = gr.Checkbox(label="🛡️ Decensor Mode", value=s.get("decensor_mode", False))

            with gr.Row():
                use_weighting = gr.Checkbox(label="⚖️ Enable Weights", value=s.get("use_weighting", False))
                aspect_ratio_sync = gr.Checkbox(label="📐 Smart Resizing", value=s.get("aspect_ratio_sync", False))

            with gr.Row():
                fusion_mode = gr.Dropdown(
                    ["Single Image", "Combine (Add + Recount)", "Replace (Full Character Swap)"],
                    value=s.get("fusion_mode", "Single Image"),
                    label="🧬 Fusion Mode"
                )
                weight_preset = gr.Dropdown(
                    ["Pony/XL (Max 1.3)", "NAI (Max 2.0)", "Custom (Manual)"],
                    value=s.get("weight_preset", "Pony/XL (Max 1.3)"),
                    label="Weight Preset"
                )
                custom_weight = gr.Slider(
                    0.5, 10.0, s.get("custom_weight", 1.5),
                    step=0.1, label="Custom Weight", visible=False
                )

            with gr.Row():
                ref_image = gr.Image(label="Image 1 (Batch/Scene Source)", type="pil", interactive=True)
                ref_image_2 = gr.Image(label="Image 2 (Character/Swap Source)", type="pil", interactive=True)

            with gr.Row():
                threshold = gr.Slider(0.0, 1.0, s.get("threshold", 0.35), label="Threshold")
                char_threshold = gr.Slider(0.0, 1.0, s.get("char_threshold", 0.85), label="Char Threshold")

            mode = gr.Radio(["Append", "Prepend", "Replace"], value=s.get("mode", "Append"), label="Prompt Mode")

            with gr.Accordion("📦 Batch Sources", open=False):
                with gr.Tabs():
                    with gr.Tab("🎨 Pixiv Gallery"):
                        pixiv_enabled = gr.Checkbox(label="🚀 Enable Pixiv", value=s.get("pixiv_enabled", False))
                        pixiv_url = gr.Textbox(label="Pixiv URL", placeholder="https://...", value=s.get("pixiv_url", ""))
                        download_only_btn = gr.Button("📥 Download Images", variant="secondary")
                        delay_timer = gr.Slider(1.0, 5.0, s.get("delay_timer", 2.0), label="Safety Delay")
                        download_status = gr.Markdown("Ready.")
                    with gr.Tab("📂 Local Folder"):
                        batch_enabled = gr.Checkbox(label="🚀 Enable Folder", value=s.get("batch_enabled", False))
                        folder_path = gr.Textbox(label="Folder Path", placeholder="E:/References", value=s.get("folder_path", ""))

                with gr.Row():
                    manual_index = gr.Number(label="Counter", value=s.get("manual_index", 0), precision=0)
                    index_mode = gr.Dropdown(["Fixed", "Increment", "Decrement", "Random"], value=s.get("index_mode", "Increment"), label="Mode")
                    reset_btn = gr.Button("♻️ Reset Counter", variant="secondary")

                with gr.Row():
                    sort_by = gr.Dropdown(["Name", "Date Modified", "File Size"], value=s.get("sort_by", "Name"), label="Sort By")
                    sort_dir = gr.Dropdown(["Ascending", "Descending"], value=s.get("sort_dir", "Ascending"), label="Sort Direction")

                log_batch = gr.Checkbox(label="📝 Log processed file (console)", value=s.get("log_batch", True))

            with gr.Accordion("⚙️ Filters", open=False):
                filter_toggles = gr.CheckboxGroup(
                    choices=["No Males", "No Females", "No Cum", "No Vaginal Sex", "No Anal", "No Oral", "No Pubic Hair"],
                    label="Quick Filters",
                    value=s.get("filter_toggles", [])
                )
                with gr.Row():
                    filter_neg_only = gr.Checkbox(label="Move to Negative", value=s.get("filter_neg_only", False))
                    filter_opposites = gr.Checkbox(label="Add Opposites", value=s.get("filter_opposites", False))
                exclude = gr.Textbox(label="Exclude Tags", value=s.get("exclude", ""), placeholder="lowres")
                replacer_list = gr.Textbox(label="find:replace", lines=3, value=s.get("replacer_list", ""))

        # --------------------------
        # QUICKSHOT UI
        # --------------------------
        with gr.Accordion("🚀 Quick-Shot Auto Prompts", open=True):
            with gr.Row():
                qs_enabled = gr.Checkbox(label="Enable Quick Prompts", value=s.get("qs_enabled", False))
                nai_mode = gr.Checkbox(label="🎨 NovelAI Weighting Mode", value=s.get("nai_mode", False))
                qs_inject_mode = gr.Dropdown(
                    ["Append", "Prepend", "Replace"],
                    value=s.get("qs_inject_mode", "Append"),
                    label="Injection Mode"
                )

            age_choices = ["None"] + list(AGE_PROMPT_MAP.keys())

            with gr.Row():
                age_group = gr.Dropdown(
                    age_choices,
                    value=s.get("age_group", "None"),
                    label="🧬 Age Group"
                )
                age_strength = gr.Slider(
                    0, 10,
                    s.get("age_strength", 0),
                    step=1,
                    label="Age Strength (0–10)"
                )
                remove_baby_props = gr.Checkbox(
                    label="🚫 Suppress Pacifier / Bib",
                    value=s.get("remove_baby_props", False)
                )

            with gr.Row():
                rating_val = gr.Slider(-5, 10, s.get("rating_val", 0), step=1, label="🔞 Rating Scale")
                view_val = gr.Slider(-5, 10, s.get("view_val", 0), step=1, label="📸 Multiple Views Scale")

            with gr.Row():
                io_val = gr.Slider(-5, 5, s.get("io_val", 0), step=1, label="🏠 Outdoor (-5) to Indoor (5)")

            with gr.Row():
                time_of_day = gr.Dropdown(
                    ["None", "Random", "dawn", "midday", "sunset", "midnight"],
                    value=s.get("time_of_day", "None"),
                    label="⏰ Time of Day"
                )
                time_weight = gr.Slider(0.0, 5.0, s.get("time_weight", 1.0), step=0.1, label="Time Weight")

            with gr.Row():
                vol_light = gr.Dropdown(
                    ["None", "Random", "from the left", "from the right", "from above", "from the side", "from below", "from behind"],
                    value=s.get("vol_light", "None"),
                    label="🔦 Light Direction"
                )
                vol_weight = gr.Slider(0.0, 5.0, s.get("vol_weight", 1.0), step=0.1, label="Light Weight")

            gr.Markdown("### 🎥 Camera & Depth")
            with gr.Row():
                viewpoint = gr.Dropdown(
                    ["None", "Random", "from above", "from below", "from side", "from behind", "from the front"],
                    value=s.get("viewpoint", "None"),
                    label="Viewpoint Angle"
                )
                viewpoint_scale = gr.Slider(0, 5, s.get("viewpoint_scale", 0), step=1, label="Viewpoint Scale (0-5)")
                dutch_angle = gr.Checkbox(label="Dutch Angle (Tilt)", value=s.get("dutch_angle", False))

            with gr.Row():
                fg_blur = gr.Slider(-5, 5, s.get("fg_blur", 0), step=1, label="🔍 FG Blur (Neg=Sharper)")
                bg_blur = gr.Slider(-5, 5, s.get("bg_blur", 0), step=1, label="🏔️ BG Blur (Neg=Sharper)")

            with gr.Row():
                qs_save = gr.Button("💾 Save Settings", variant="secondary")
                save_status = gr.Markdown("")

        comp_list = [
            enabled, allow_nsfw, batch_enabled, folder_path, manual_index, index_mode, ref_image,
            threshold, char_threshold, mode, exclude, pixiv_enabled, pixiv_url, delay_timer,
            use_weighting, weight_preset, aspect_ratio_sync, custom_weight, replacer_list,
            filter_toggles, filter_neg_only, filter_opposites, char_only, decensor_mode,
            ref_image_2, fusion_mode,

            sort_by, sort_dir, log_batch,

            qs_enabled, nai_mode, qs_inject_mode,
            rating_val, io_val, view_val,
            time_of_day, time_weight, vol_light, vol_weight,
            viewpoint, viewpoint_scale, dutch_angle, fg_blur, bg_blur,

            age_group, age_strength, remove_baby_props,
        ]

        weight_preset.change(
            fn=lambda p: gr.update(visible=(p == "Custom (Manual)")),
            inputs=[weight_preset],
            outputs=[custom_weight]
        )

        download_only_btn.click(fn=self.fetch_pixiv_images, inputs=[pixiv_url, delay_timer], outputs=[download_status])
        qs_save.click(fn=self.save_all_settings, inputs=comp_list, outputs=[save_status])
        reset_btn.click(fn=self.reset_counter, outputs=[manual_index])

        return comp_list

    # ======================================================
    # Settings IO
    # ======================================================
    def save_all_settings(self, *args):
        keys = [
            "enabled", "allow_nsfw", "batch_enabled", "folder_path", "manual_index", "index_mode", "ref_image",
            "threshold", "char_threshold", "mode", "exclude", "pixiv_enabled", "pixiv_url", "delay_timer",
            "use_weighting", "weight_preset", "aspect_ratio_sync", "custom_weight", "replacer_list",
            "filter_toggles", "filter_neg_only", "filter_opposites", "char_only", "decensor_mode",
            "ref_image_2", "fusion_mode",

            "sort_by", "sort_dir", "log_batch",

            "qs_enabled", "nai_mode", "qs_inject_mode",
            "rating_val", "io_val", "view_val",
            "time_of_day", "time_weight", "vol_light", "vol_weight",
            "viewpoint", "viewpoint_scale", "dutch_angle", "fg_blur", "bg_blur",

            "age_group", "age_strength", "remove_baby_props",
        ]

        def _safe(v):
            if isinstance(v, Image.Image):
                return None
            return v

        settings = {k: _safe(args[i]) for i, k in enumerate(keys) if i < len(args)}
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return "✅ Settings Saved!"

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    # ======================================================
    # Pixiv fetch
    # ======================================================
    def fetch_pixiv_images(self, url, delay=2.0):
        if not url:
            return "❌ No URL"
        phpid = getattr(shared.opts, "wd_tagger_pixiv_phpid", "")
        match = re.search(r"artworks/(\d+)", url)
        if not match:
            return "❌ Invalid URL"
        illust_id = match.group(1)
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://www.pixiv.net/en/artworks/{illust_id}",
            "Cookie": f"PHPSESSID={phpid}"
        }
        try:
            data = requests.get(f"https://www.pixiv.net/ajax/illust/{illust_id}/pages", headers=headers, timeout=10).json()
            tmp_dir = os.path.join(scripts.basedir(), "outputs", "pixiv_cache", illust_id)
            os.makedirs(tmp_dir, exist_ok=True)
            self.cached_target_files = []
            for i, page in enumerate(data["body"]):
                save_path = os.path.join(tmp_dir, f"p{i}.jpg")
                if not os.path.exists(save_path):
                    with open(save_path, "wb") as f:
                        f.write(requests.get(page["urls"]["original"], headers=headers, timeout=15).content)
                    time.sleep(float(delay))
                self.cached_target_files.append(save_path)
            return f"✅ {len(self.cached_target_files)} images ready."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    # ======================================================
    # WD14 inference
    # ======================================================
    def run_inference(self, img, aspect_ratio_sync=False):
        if img is None or self.model is None:
            return None

        img_arr = np.array(img.convert("RGB")).astype(np.float32)[:, :, ::-1]  # RGB->BGR
        h, w = img_arr.shape[:2]

        ratio = 448 / max(h, w)
        resized = cv2.resize(img_arr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

        canvas = np.full((448, 448, 3), 255, dtype=np.float32)
        y0 = (448 - resized.shape[0]) // 2
        x0 = (448 - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized

        inp = np.expand_dims(canvas, axis=0)
        return self.model.run(None, {self.model.get_inputs()[0].name: inp})[0][0]

    def _clamp_weight_xl(self, w):
        return max(0.0, min(1.5, float(w)))

    def _maybe_weight(self, text, w, nai_mode):
        if w is None:
            return text
        w = float(w)
        if nai_mode:
            return f"({text}:{w:.2f})"
        w = self._clamp_weight_xl(w)
        return f"({text}:{w:.2f})"

    # ======================================================
    # ReForge-safe base index resolve (global index)
    # ======================================================
    def _resolve_base_index(self, p, batch_prompts):
        bsz = len(batch_prompts)
        allp = getattr(p, "all_prompts", None)

        if isinstance(allp, list) and allp and len(allp) >= bsz:
            start = max(0, int(self._search_pos or 0))
            limit = len(allp) - bsz
            for i in range(start, limit + 1):
                if allp[i:i + bsz] == batch_prompts:
                    self._search_pos = i + bsz
                    self._global_cursor = max(self._global_cursor, i + bsz)
                    return i
            for i in range(0, limit + 1):
                if allp[i:i + bsz] == batch_prompts:
                    self._search_pos = i + bsz
                    self._global_cursor = max(self._global_cursor, i + bsz)
                    return i

        # fallback: sequential cursor
        i = int(self._global_cursor or 0)
        self._global_cursor = i + bsz
        return i

    # ======================================================
    # CONFIG STORE (NO TAGGING HERE)
    # ======================================================
    def process(
        self,
        p,
        enabled, allow_nsfw, batch_enabled, folder_path, manual_index, index_mode, ref_image,
        threshold, char_threshold, mode, exclude, pixiv_enabled, pixiv_url, delay_timer,
        use_weighting, weight_preset, aspect_ratio_sync, custom_weight, replacer_list,
        filter_toggles, filter_neg_only, filter_opposites, char_only, decensor_mode,
        ref_image_2, fusion_mode,
        sort_by, sort_dir, log_batch,
        qs_enabled, nai_mode, qs_inject_mode,
        rating_val, io_val, view_val,
        time_of_day, time_weight, vol_light, vol_weight,
        viewpoint, viewpoint_scale, dutch_angle, fg_blur, bg_blur,
        age_group, age_strength, remove_baby_props
    ):
        cfg = dict(
            enabled=bool(enabled),
            allow_nsfw=bool(allow_nsfw),
            batch_enabled=bool(batch_enabled),
            folder_path=str(folder_path or ""),
            manual_index=int(manual_index or 0),
            index_mode=str(index_mode or "Increment"),
            ref_image=ref_image if isinstance(ref_image, Image.Image) else None,
            ref_image_2=ref_image_2 if isinstance(ref_image_2, Image.Image) else None,
            threshold=float(threshold),
            char_threshold=float(char_threshold),
            mode=str(mode or "Append"),
            exclude=str(exclude or ""),
            pixiv_enabled=bool(pixiv_enabled),
            pixiv_url=str(pixiv_url or ""),
            delay_timer=float(delay_timer or 2.0),
            use_weighting=bool(use_weighting),
            weight_preset=str(weight_preset or "Pony/XL (Max 1.3)"),
            aspect_ratio_sync=bool(aspect_ratio_sync),
            custom_weight=float(custom_weight or 1.5),
            replacer_list=str(replacer_list or ""),
            filter_toggles=list(filter_toggles or []),
            filter_neg_only=bool(filter_neg_only),
            filter_opposites=bool(filter_opposites),
            char_only=bool(char_only),
            decensor_mode=bool(decensor_mode),
            fusion_mode=str(fusion_mode or "Single Image"),
            sort_by=str(sort_by or "Name"),
            sort_dir=str(sort_dir or "Ascending"),
            log_batch=bool(log_batch),

            qs_enabled=bool(qs_enabled),
            nai_mode=bool(nai_mode),
            qs_inject_mode=str(qs_inject_mode or "Append"),
            rating_val=rating_val,
            io_val=io_val,
            view_val=view_val,
            time_of_day=str(time_of_day or "None"),
            time_weight=float(time_weight or 1.0),
            vol_light=str(vol_light or "None"),
            vol_weight=float(vol_weight or 1.0),
            viewpoint=str(viewpoint or "None"),
            viewpoint_scale=int(viewpoint_scale or 0),
            dutch_angle=bool(dutch_angle),
            fg_blur=fg_blur,
            bg_blur=bg_blur,
            age_group=str(age_group or "None"),
            age_strength=int(age_strength or 0),
            remove_baby_props=bool(remove_baby_props),
        )

        # Build files list ONCE per job and store into cfg
        files = []
        if cfg["pixiv_enabled"] and self.cached_target_files:
            files = list(self.cached_target_files)
        elif cfg["batch_enabled"] and cfg["folder_path"]:
            try:
                files = [
                    os.path.join(cfg["folder_path"], f)
                    for f in os.listdir(cfg["folder_path"])
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ]
            except Exception:
                files = []
        files = self.sort_files(files, cfg["sort_by"], cfg["sort_dir"])
        cfg["_files"] = files
        cfg["_num_files"] = len(files)
        cfg["_fusion_tags2_pos"] = None  # lazy computed later

        # Counter signature (resets offset when batch source changes)
        sig = (
            cfg["index_mode"], cfg["manual_index"],
            self._files_signature(files, cfg["pixiv_enabled"], cfg["pixiv_url"], cfg["batch_enabled"], cfg["folder_path"], cfg["sort_by"], cfg["sort_dir"])
        )
        if sig != self._counter_signature:
            self.batch_offset = 0
            self._counter_signature = sig

        # Job signature (resets global cursors + exports)
        job_sig = (
            cfg["pixiv_enabled"], cfg["pixiv_url"],
            cfg["batch_enabled"], cfg["folder_path"],
            cfg["index_mode"], cfg["manual_index"],
            cfg["sort_by"], cfg["sort_dir"],
            getattr(p, "seed", None),
            getattr(p, "n_iter", None),
            getattr(p, "batch_size", None),
            len(getattr(p, "all_prompts", []) or []),
        )
        if job_sig != self._job_sig:
            self._job_sig = job_sig
            self._search_pos = 0
            self._global_cursor = 0

            # fresh exports dicts for the whole job
            shared.wd14_batch_image_paths = {}
            shared.wd14_batch_tag_payloads = {}
            shared.wd14_batch_neg_tags = {}

        # Slot export for JoyCaption
        if isinstance(cfg["ref_image"], Image.Image):
            shared.wd14_slot_image_pil = cfg["ref_image"]
            try:
                p._wd14_slot_image_pil = cfg["ref_image"]
            except Exception:
                pass
        else:
            if not hasattr(shared, "wd14_slot_image_pil"):
                shared.wd14_slot_image_pil = None

        # Bind this p to shared exports (important if p is cloned later)
        try:
            p._wd14_batch_image_paths = shared.wd14_batch_image_paths
            p._wd14_batch_tag_payloads = shared.wd14_batch_tag_payloads
            p._wd14_batch_neg_tags = shared.wd14_batch_neg_tags
        except Exception:
            pass

        self._active_cfg = cfg
        print("✅ [WD14] process(): cfg stored. Tagging will run in process_batch().")
        return

    # ======================================================
    # BATCH-SAFE TAGGING (THIS IS THE FIX)
    # ======================================================
    def process_batch(self, p, *args, **kwargs):
        cfg = self._active_cfg
        if not isinstance(cfg, dict):
            return

        # Locate prompts list for this batch
        prompts = kwargs.get("prompts", None)

        if prompts is None:
            # scan args for a list[str]
            for a in args:
                if isinstance(a, list) and a and all(isinstance(x, str) for x in a):
                    prompts = a
                    break

        if prompts is None:
            # fallback to p.prompts if available
            prompts = getattr(p, "prompts", None)

        if not (isinstance(prompts, list) and prompts and all(isinstance(x, str) for x in prompts)):
            return

        bsz = len(prompts)

        # Locate negative prompts if possible
        negative_prompts = kwargs.get("negative_prompts", None)
        if not (isinstance(negative_prompts, list) and len(negative_prompts) == bsz):
            negative_prompts = getattr(p, "negative_prompts", None)

        if not (isinstance(negative_prompts, list) and len(negative_prompts) == bsz):
            base_neg = getattr(p, "negative_prompt", "") or ""
            negative_prompts = [base_neg] * bsz

        base_index = self._resolve_base_index(p, prompts)
        files = cfg.get("_files", []) or []
        num_files = int(cfg.get("_num_files", 0) or 0)

        # Ensure exports dicts exist
        if not hasattr(shared, "wd14_batch_image_paths") or not isinstance(shared.wd14_batch_image_paths, dict):
            shared.wd14_batch_image_paths = {}
        if not hasattr(shared, "wd14_batch_tag_payloads") or not isinstance(shared.wd14_batch_tag_payloads, dict):
            shared.wd14_batch_tag_payloads = {}
        if not hasattr(shared, "wd14_batch_neg_tags") or not isinstance(shared.wd14_batch_neg_tags, dict):
            shared.wd14_batch_neg_tags = {}

        try:
            p._wd14_batch_image_paths = shared.wd14_batch_image_paths
            p._wd14_batch_tag_payloads = shared.wd14_batch_tag_payloads
            p._wd14_batch_neg_tags = shared.wd14_batch_neg_tags
        except Exception:
            pass

        # --------------------------
        # QuickShot applies per batch
        # --------------------------
        if cfg.get("qs_enabled", False):
            for i in range(bsz):
                prompts[i], negative_prompts[i] = apply_quickshot(
                    prompts[i],
                    negative_prompts[i],
                    inject_mode=cfg["qs_inject_mode"],
                    nai_mode=cfg["nai_mode"],
                    rating_val=cfg["rating_val"],
                    io_val=cfg["io_val"],
                    view_val=cfg["view_val"],
                    time_of_day=cfg["time_of_day"],
                    time_weight=cfg["time_weight"],
                    vol_light=cfg["vol_light"],
                    vol_weight=cfg["vol_weight"],
                    viewpoint=cfg["viewpoint"],
                    viewpoint_scale=cfg["viewpoint_scale"],
                    dutch_angle=cfg["dutch_angle"],
                    fg_blur=cfg["fg_blur"],
                    bg_blur=cfg["bg_blur"],
                    age_group=cfg["age_group"],
                    age_strength=cfg["age_strength"],
                    remove_baby_props=cfg["remove_baby_props"],
                    maybe_weight=self._maybe_weight,
                    clamp_weight_xl=self._clamp_weight_xl,
                )

        # If WD14 disabled, still push QuickShot changes into p and exit
        if not cfg.get("enabled", False):
            try:
                p.prompts = prompts
                p.negative_prompts = negative_prompts
                p.prompt = prompts[0]
                p.negative_prompt = negative_prompts[0]
            except Exception:
                pass
            return

        # --------------------------
        # Load WD14 model
        # --------------------------
        onnx_path = getattr(shared.opts, "wd_tagger_model_path", "")
        csv_path = getattr(shared.opts, "wd_tagger_csv_path", "")

        if not onnx_path or not csv_path or not os.path.exists(onnx_path) or not os.path.exists(csv_path):
            print("⚠️ [WD14] Missing model_path/csv_path. Set them in Settings → WD14 Tagger.")
            return

        if self.model is None or self.tags is None:
            self.model = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.tags = pd.read_csv(csv_path)

        # --------------------------
        # Weight cap
        # --------------------------
        try:
            max_w = 1.3 if "Pony" in cfg["weight_preset"] else (2.0 if "NAI" in cfg["weight_preset"] else float(cfg["custom_weight"]))
        except Exception:
            max_w = 1.3

        # --------------------------
        # Excludes + filters
        # --------------------------
        ex_list = [x.strip().lower() for x in (cfg["exclude"] or "").split(",") if x.strip()]
        if "No Pubic Hair" in (cfg.get("filter_toggles") or []):
            ex_list.extend(self.pubic_hair_tags)

        filter_map = {
            "No Males": ["male", "boy", "man", "1boy", "penis"],
            "No Females": ["female", "girl", "woman", "loli", "pussy", "vagina", "breasts"],
            "No Cum": ["cum", "semen"],
            "No Vaginal Sex": ["pussy", "vagina", "vaginal sex", "creampie"],
            "No Anal": ["anal", "asshole"],
            "No Oral": ["oral", "blowjob"],
        }
        for t in (cfg.get("filter_toggles") or []):
            ex_list.extend(filter_map.get(t, []))

        # --------------------------
        # Replacements
        # --------------------------
        replacements = {}
        for line in (cfg.get("replacer_list") or "").split("\n"):
            if ":" in line:
                a, b = line.split(":", 1)
                replacements[a.strip().lower()] = b.strip().lower()

        # --------------------------
        # Fusion tags (lazy compute once)
        # --------------------------
        tags2_pos = cfg.get("_fusion_tags2_pos", None)
        if tags2_pos is None:
            tags2_pos = []
            if cfg.get("fusion_mode") != "Single Image" and isinstance(cfg.get("ref_image_2"), Image.Image):
                probs2 = self.run_inference(cfg["ref_image_2"], aspect_ratio_sync=cfg.get("aspect_ratio_sync", False))
                if probs2 is not None:
                    for j, prob in enumerate(probs2):
                        row = self.tags.iloc[j]
                        name = str(row["name"]).replace("_", " ").lower()
                        cat = int(row["category"])
                        if cat == 9 or prob < (cfg["char_threshold"] if cat == 4 else cfg["threshold"]):
                            continue
                        is_char = (any(k in name for k in self.char_design_keywords) or cat == 4)
                        p_tag = replacements.get(name, name)
                        tag_str = f"({p_tag}:{1.0 + (prob * (max_w - 1.0)):.2f})" if cfg.get("use_weighting") else p_tag

                        if "Combine" in cfg["fusion_mode"]:
                            tags2_pos.append(tag_str)
                        elif "Replace" in cfg["fusion_mode"] and is_char:
                            tags2_pos.append(tag_str)

            cfg["_fusion_tags2_pos"] = tags2_pos

        # --------------------------
        # Main loop (THIS BATCH ONLY)
        # --------------------------
        base_counter = int(cfg.get("manual_index", 0) or 0)
        for bi in range(bsz):
            gi = base_index + bi  # GLOBAL prompt index (this is what JoyCaption uses)

            img1 = None
            chosen_path = None
            idx = None

            if num_files > 0:
                # ✅ Uses gi so it stays consistent across batches
                if cfg["index_mode"] == "Increment":
                    idx = (base_counter + self.batch_offset + gi) % num_files
                elif cfg["index_mode"] == "Decrement":
                    idx = (base_counter - (self.batch_offset + gi)) % num_files
                elif cfg["index_mode"] == "Random":
                    idx = random.randint(0, num_files - 1)
                else:
                    idx = base_counter % num_files

                chosen_path = files[idx]
                try:
                    with Image.open(chosen_path) as im:
                        img1 = im.convert("RGB")
                except Exception:
                    img1 = None
            else:
                img1 = cfg.get("ref_image") if isinstance(cfg.get("ref_image"), Image.Image) else None

            # ✅ EXPORT EXACT IMAGE PATH USED FOR THIS GLOBAL INDEX
            shared.wd14_batch_image_paths[gi] = chosen_path if chosen_path else None

            if cfg.get("log_batch", True):
                if chosen_path:
                    print(f"[WD14] {os.path.basename(chosen_path)} ({bi+1}/{bsz}) gidx={gi} idx={idx} mode={cfg['index_mode']}")
                else:
                    print(f"[WD14] ref_image ({bi+1}/{bsz}) gidx={gi} mode={cfg['index_mode']}")

            if img1 is None:
                continue

            probs1 = self.run_inference(img1, aspect_ratio_sync=cfg.get("aspect_ratio_sync", False))
            if probs1 is None:
                continue

            final_pos, final_neg, img1_char_tags = [], [], []

            for j, prob in enumerate(probs1):
                row = self.tags.iloc[j]
                name = str(row["name"]).replace("_", " ").lower()
                cat = int(row["category"])
                if cat == 9 or prob < (cfg["char_threshold"] if cat == 4 else cfg["threshold"]):
                    continue

                if cfg.get("decensor_mode") and any(ck in name for ck in self.censor_keywords):
                    final_neg.append(name)
                    continue

                if any(ex in name for ex in ex_list):
                    if cfg.get("filter_neg_only"):
                        final_neg.append(name)
                    continue

                is_char = (any(k in name for k in self.char_design_keywords) or cat == 4)
                if cfg.get("char_only") and not is_char:
                    continue

                p_tag = replacements.get(name, name)
                tag_str = f"({p_tag}:{1.0 + (prob * (max_w - 1.0)):.2f})" if cfg.get("use_weighting") else p_tag

                if is_char:
                    img1_char_tags.append(tag_str)
                final_pos.append(tag_str)

            # Fusion application
            if tags2_pos:
                if "Combine" in cfg["fusion_mode"]:
                    final_pos = list(dict.fromkeys(final_pos + tags2_pos))
                elif "Replace" in cfg["fusion_mode"]:
                    final_pos = [t for t in final_pos if t not in img1_char_tags] + tags2_pos

            tag_payload = ", ".join(final_pos)

            # Export debug payloads (global index)
            shared.wd14_batch_tag_payloads[gi] = tag_payload
            shared.wd14_batch_neg_tags[gi] = ", ".join(sorted(set(final_neg))) if final_neg else ""

            # Inject into batch prompts
            if cfg["mode"] == "Append":
                if tag_payload:
                    prompts[bi] += f", {tag_payload}"
            elif cfg["mode"] == "Prepend":
                if tag_payload:
                    prompts[bi] = f"{tag_payload}, {prompts[bi]}"
            else:
                if tag_payload:
                    prompts[bi] = tag_payload

            if final_neg:
                negative_prompts[bi] += f", {', '.join(sorted(set(final_neg)))}"

            # Also update global arrays if present
            try:
                if hasattr(p, "all_prompts") and isinstance(p.all_prompts, list) and 0 <= gi < len(p.all_prompts):
                    p.all_prompts[gi] = prompts[bi]
                if hasattr(p, "all_negative_prompts") and isinstance(p.all_negative_prompts, list) and 0 <= gi < len(p.all_negative_prompts):
                    p.all_negative_prompts[gi] = negative_prompts[bi]
            except Exception:
                pass

        # Advance counter across runs (keeps your “continuous folder indexing” behavior)
        if num_files > 0 and cfg["index_mode"] in ("Increment", "Decrement"):
            self.batch_offset = (self.batch_offset + bsz) % num_files

        # Push back to p
        try:
            p.prompts = prompts
            p.negative_prompts = negative_prompts
            p.prompt = prompts[0]
            p.negative_prompt = negative_prompts[0]
        except Exception:
            pass

        print(f"✅ [WD14] process_batch(): injected {bsz} prompts (base_index={base_index}).")

        return
