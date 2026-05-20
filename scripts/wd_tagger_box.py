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
from wd_quickshot_helper import apply_quickshot, AGE_PROMPT_MAP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from nai_char_prompt_builder import inject_global_before_chars, merge_chars_into_prompt

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
    shared.opts.add_option(
        "wd_tagger_pixiv_phpid",
        shared.OptionInfo("", "Pixiv PHPSESSID (for Pixiv download)", section=section),
    )
    shared.opts.add_option(
        "wd_tagger_yolo_path",
        shared.OptionInfo("", "YOLO model for multi-char detection (.pt path; blank = auto-download yolov8n.pt)", section=section),
    )

script_callbacks.on_ui_settings(on_ui_settings)


# Capture the main prompt textboxes so the CHAR button can write straight into them (no copy-paste).
# on_after_component fires for every component as the UI is built; the toprow prompt is created
# before the script UIs, so these are populated by the time ui() runs.
_MAIN_PROMPT_BOXES = {}

def _capture_main_prompt(component, **kwargs):
    eid = getattr(component, "elem_id", None)
    if eid in ("txt2img_prompt", "img2img_prompt"):
        _MAIN_PROMPT_BOXES[eid.split("_")[0]] = component

script_callbacks.on_after_component(_capture_main_prompt)


def _detect_char_token(prompt: str) -> str:
    """Auto-detect which CHAR style the prompt is using."""
    p = prompt or ""
    if "[CHAR" in p:
        return "CHAR:"
    if "CHAR:" in p:
        return "CHAR:"
    # default to NovelAI block style
    return "CHAR:"


class WDTaggerBox(scripts.Script):
    def __init__(self):
        self._nai_patched = False
        self.model = None
        self.tags = None
        self._onnx_path = None
        self._csv_path = None

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

        # YOLO person detector (lazy-loaded for multi-character CHAR blocks)
        self._yolo = None
        self._yolo_path = None

    def title(self):
        return "WD14 Tagger (Integrated) v2.0.7 - ReForge Batch Fix + JoyCaption Bridge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def before_process(self, p, *args, **kwargs):
        # Apply NAI compat patch once per session so NAI reads updated prompt lists.
        if getattr(self, "_nai_patched", False):
            return

        # Process-wide guard: if any script (or a prior generation) already patched,
        # don't reload the patch module again. self._nai_patched resets when ReForge
        # recreates the script instance, so this shared flag is what actually sticks.
        if getattr(shared, "_nai_compat_patch_done", False):
            self._nai_patched = True
            return

        try:
            import importlib.util
            import os

            here = os.path.dirname(os.path.abspath(__file__))  # .../scripts
            patch_path = os.path.join(here, "patch", "nai_compat_patch.py")  # .../scripts/patch/nai_compat_patch.py

            if not os.path.isfile(patch_path):
                print(f"⚠️ [WD14] NAI compat patch not found: {patch_path}")
                self._nai_patched = True
                return

            spec = importlib.util.spec_from_file_location("wd14_nai_compat_patch", patch_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            ok = bool(mod.apply_nai_patch())
            self._nai_patched = True
            print(f"✅ [WD14] NAI compat patch {'enabled' if ok else 'not applied'}.")
        except Exception as e:
            self._nai_patched = True
            print(f"⚠️ [WD14] NAI compat patch error: {e}")
    

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

            with gr.Accordion("🧬 NAI CHAR Blocks (from WD14)", open=False):
                gr.Markdown(
                    "Tag the loaded image(s) with WD14 and split the **character traits** into `CHAR:` slots. "
                    "WD14 only — no JoyCaption. **➕ Add CHAR Blocks to Prompt** writes them straight into "
                    "your prompt (no copy-paste); **Build CHAR Blocks** just fills the box below to review/copy.\n\n"
                    "- **Auto-detect OFF:** Image 1 → CHAR1, Image 2 → CHAR2 (one image per character).\n"
                    "- **Auto-detect ON:** finds every person in **Image 1** with YOLO, crops each, and makes a "
                    "CHAR slot per person (up to 6). Scene/global tags stay above the blocks."
                )
                with gr.Row():
                    multi_char = gr.Checkbox(label="🔍 Auto-detect characters (YOLO, multi-char from Image 1)", value=False)
                    det_conf = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Detection Confidence")
                with gr.Row():
                    build_char_btn = gr.Button("🧍 Build CHAR Blocks (to box)", variant="secondary")
                    send_char_btn = gr.Button("➕ Add CHAR Blocks to Prompt", variant="primary")
                char_output = gr.Textbox(
                    label="CHAR Output (copy into prompt)",
                    lines=6, interactive=True, show_copy_button=True
                )

            with gr.Accordion("📦 Batch Sources", open=False):
                with gr.Tabs():
                    with gr.TabItem("🎨 Pixiv Gallery"):
                        pixiv_enabled = gr.Checkbox(label="🚀 Enable Pixiv", value=s.get("pixiv_enabled", False))
                        pixiv_url = gr.Textbox(label="Pixiv URL", placeholder="https://...", value=s.get("pixiv_url", ""))
                        download_only_btn = gr.Button("📥 Download Images", variant="secondary")
                        delay_timer = gr.Slider(1.0, 5.0, s.get("delay_timer", 2.0), label="Safety Delay")
                        download_status = gr.Markdown("Ready.")
                    with gr.TabItem("📂 Local Folder"):
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
        with gr.Accordion("🚀 Quick-Shot Auto Prompts", open=False):
            with gr.Row():
                qs_enabled = gr.Checkbox(label="Enable Quick Prompts", value=s.get("qs_enabled", False))
                nai_mode = gr.Checkbox(label="🎨 NovelAI Weighting Mode", value=s.get("nai_mode", False))
                nai_char_mode = gr.Checkbox(label="🧍 Enable NovelAI CHAR blocks", value=s.get("nai_char_mode", False))
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
            use_weighting, weight_preset, custom_weight, replacer_list,
            filter_toggles, filter_neg_only, filter_opposites, char_only, decensor_mode,
            ref_image_2, fusion_mode,
            sort_by, sort_dir, log_batch,
            qs_enabled, nai_mode, nai_char_mode, qs_inject_mode,
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
        build_char_btn.click(
            fn=self.build_char_blocks,
            inputs=[ref_image, ref_image_2, threshold, char_threshold, exclude, replacer_list,
                    filter_toggles, char_only, use_weighting, weight_preset, custom_weight,
                    multi_char, det_conf],
            outputs=[char_output],
        )
        _main_prompt = _MAIN_PROMPT_BOXES.get("img2img" if is_img2img else "txt2img")
        if _main_prompt is not None:
            send_char_btn.click(
                fn=self.add_char_to_prompt,
                inputs=[_main_prompt, ref_image, ref_image_2, threshold, char_threshold, exclude, replacer_list,
                        filter_toggles, char_only, use_weighting, weight_preset, custom_weight, multi_char, det_conf],
                outputs=[_main_prompt],
            )
        else:
            # Fallback: prompt box wasn't captured -> still do something useful (fill the box).
            send_char_btn.click(
                fn=self.build_char_blocks,
                inputs=[ref_image, ref_image_2, threshold, char_threshold, exclude, replacer_list,
                        filter_toggles, char_only, use_weighting, weight_preset, custom_weight,
                        multi_char, det_conf],
                outputs=[char_output],
            )

        # Let THIS plugin's Save (tagger_quickshot_settings.json) own these defaults instead of
        # Forge's ui-config.json, which otherwise tracks every labeled component and overrides the
        # plugin's saved values on restart (resetting toggles). Exempt all saved settings + the
        # CHAR-builder controls.
        for _c in comp_list + [multi_char, det_conf, char_output, build_char_btn, send_char_btn]:
            try:
                _c.do_not_save_to_config = True
            except Exception:
                pass

        return comp_list

    # ======================================================
    # Settings IO
    # ======================================================
    def save_all_settings(self, *args):
        keys = [
            "enabled", "allow_nsfw", "batch_enabled", "folder_path", "manual_index", "index_mode", "ref_image",
            "threshold", "char_threshold", "mode", "exclude", "pixiv_enabled", "pixiv_url", "delay_timer",
            "use_weighting", "weight_preset", "custom_weight", "replacer_list",
            "filter_toggles", "filter_neg_only", "filter_opposites", "char_only", "decensor_mode",
            "ref_image_2", "fusion_mode",
            "sort_by", "sort_dir", "log_batch",
            "qs_enabled", "nai_mode", "nai_char_mode", "qs_inject_mode",
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
            data = requests.get(
                f"https://www.pixiv.net/ajax/illust/{illust_id}/pages",
                headers=headers,
                timeout=10
            ).json()
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
    def run_inference(self, img):
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

        # ✅ FIX: normalize to 0..1 (you were missing this)
        

        inp = np.expand_dims(canvas, axis=0)
        return self.model.run(None, {self.model.get_inputs()[0].name: inp})[0][0]

    def _detect_people(self, pil_img, conf=0.35):
        """Detect people in a PIL image with YOLO (ultralytics). Returns (boxes, error):
        boxes = list of (x1,y1,x2,y2) ints sorted left-to-right; error is None on success."""
        if not isinstance(pil_img, Image.Image):
            return None, "no image"
        try:
            from ultralytics import YOLO
        except Exception as e:
            return None, f"ultralytics not available ({e})"
        yolo_path = getattr(shared.opts, "wd_tagger_yolo_path", "") or "yolov8n.pt"
        if self._yolo is None or self._yolo_path != yolo_path:
            try:
                self._yolo = YOLO(yolo_path)
                self._yolo_path = yolo_path
            except Exception as e:
                self._yolo = None
                return None, f"could not load YOLO '{yolo_path}' ({e})"
        src = pil_img.convert("RGB")
        try:
            results = self._yolo.predict(source=src, conf=float(conf), classes=[0], verbose=False)
        except Exception:
            # model may not use COCO class 0 = person; retry without the class filter
            try:
                results = self._yolo.predict(source=src, conf=float(conf), verbose=False)
            except Exception as e:
                return None, f"detection error ({e})"
        boxes = []
        for r in results:
            b = getattr(r, "boxes", None)
            if b is None:
                continue
            try:
                xyxy = b.xyxy.cpu().numpy()
            except Exception:
                continue
            for row in xyxy:
                x1, y1, x2, y2 = [int(v) for v in row[:4]]
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        boxes.sort(key=lambda bb: bb[0])  # left-to-right => CHAR1, CHAR2, ...
        return boxes, None

    def build_char_blocks(self, ref_image, ref_image_2, threshold, char_threshold, exclude, replacer_list,
                          filter_toggles, char_only, use_weighting, weight_preset, custom_weight,
                          multi_char=False, det_conf=0.35, base_prompt=""):
        """Manual button: tag the loaded image(s) with WD14 and split character traits into
        CHAR: slots (Image 1 -> CHAR1, Image 2 -> CHAR2). WD14 only — no JoyCaption, no
        auto-inject. Returns a string to copy into a NovelAI prompt."""
        onnx_path = getattr(shared.opts, "wd_tagger_model_path", "")
        csv_path = getattr(shared.opts, "wd_tagger_csv_path", "")
        if not onnx_path or not csv_path or not os.path.exists(onnx_path) or not os.path.exists(csv_path):
            return "⚠️ Set WD14 Model Path (.onnx) and Tags Path (.csv) in Settings → WD14 Tagger first."
        if not isinstance(ref_image, Image.Image):
            return "⚠️ Load an image into 'Image 1' first."

        # Load model + tags (reuse cached if already loaded)
        if self.model is None or self._onnx_path != onnx_path:
            try:
                self.model = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception:
                self.model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._onnx_path = onnx_path
        if self.tags is None or self._csv_path != csv_path:
            try:
                self.tags = pd.read_csv(csv_path)
                self._csv_path = csv_path
            except Exception as e:
                return f"⚠️ Failed to read tags.csv: {e}"
        if "name" not in self.tags.columns or "category" not in self.tags.columns:
            return "⚠️ tags.csv missing required columns: name/category"

        # Same weight cap / filters / replacements rules as the auto path
        try:
            max_w = 1.3 if "Pony" in (weight_preset or "") else (2.0 if "NAI" in (weight_preset or "") else float(custom_weight or 1.5))
        except Exception:
            max_w = 1.3

        ex_list = [x.strip().lower() for x in (exclude or "").split(",") if x.strip()]
        if "No Pubic Hair" in (filter_toggles or []):
            ex_list.extend(self.pubic_hair_tags)
        filter_map = {
            "No Males": ["male", "boy", "man", "1boy", "penis"],
            "No Females": ["female", "girl", "woman", "loli", "pussy", "vagina", "breasts"],
            "No Cum": ["cum", "semen"],
            "No Vaginal Sex": ["pussy", "vagina", "vaginal sex", "creampie"],
            "No Anal": ["anal", "asshole"],
            "No Oral": ["oral", "blowjob"],
        }
        for t in (filter_toggles or []):
            ex_list.extend(filter_map.get(t, []))

        replacements = {}
        for line in (replacer_list or "").split("\n"):
            if ":" in line:
                a, b = line.split(":", 1)
                replacements[a.strip().lower()] = b.strip().lower()

        thr = float(threshold)
        cthr = float(char_threshold)

        def _tag(img):
            pos, chars = [], []
            probs = self.run_inference(img)
            if probs is None:
                return pos, chars
            for j, prob in enumerate(probs):
                row = self.tags.iloc[j]
                name = str(row["name"]).replace("_", " ").lower()
                cat = int(row["category"])
                if cat == 9 or prob < (cthr if cat == 4 else thr):
                    continue
                if any(ex in name for ex in ex_list):
                    continue
                is_char = (any(k in name for k in self.char_design_keywords) or cat == 4)
                if bool(char_only) and not is_char:
                    continue
                p_tag = replacements.get(name, name)
                tag_str = f"({p_tag}:{1.0 + (prob * (max_w - 1.0)):.2f})" if bool(use_weighting) else p_tag
                if is_char:
                    chars.append(tag_str)
                pos.append(tag_str)
            return pos, chars

        if multi_char:
            # Auto multi-character: detect each person in Image 1, crop, tag each crop -> one CHAR per person.
            boxes, err = self._detect_people(ref_image, det_conf)
            if err:
                return ("⚠️ Multi-char detection failed: " + err +
                        "\n(Tip: set a YOLO .pt path in Settings → WD14 Tagger, or uncheck Auto-detect.)")
            if not boxes:
                return "⚠️ No people detected. Lower Detection Confidence, or uncheck Auto-detect to use Image 1/Image 2."
            W, H = ref_image.size
            char_blocks, char_set = [], set()
            for (x1, y1, x2, y2) in boxes[:6]:
                px = int((x2 - x1) * 0.06)
                py = int((y2 - y1) * 0.06)
                crop = ref_image.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))
                _cpos, cch = _tag(crop)
                if cch:
                    char_blocks.append(cch)
                    char_set.update(cch)
            if not char_blocks:
                return "⚠️ Detected people but found no character traits — lower the Char Threshold and retry."
            pos_full, char_full = _tag(ref_image)
            global_tags = [t for t in pos_full if t not in set(char_full) and t not in char_set]
        else:
            # Manual: Image 1 -> CHAR1, Image 2 -> CHAR2.
            pos1, char1 = _tag(ref_image)
            char_blocks = [char1] if char1 else []
            char_set = set(char1)
            if isinstance(ref_image_2, Image.Image):
                _pos2, char2 = _tag(ref_image_2)
                if char2:
                    char_blocks.append(char2)
                    char_set.update(char2)
            global_tags = [t for t in pos1 if t not in char_set]

        out = inject_global_before_chars(base_prompt or "", ", ".join(global_tags), mode="Append", token="CHAR:")
        out = merge_chars_into_prompt(out, char_blocks, token="CHAR:", max_chars=6)
        out = (out or "").strip()
        if not out:
            return "⚠️ No tags passed the thresholds. Lower the Threshold and try again."
        return out

    def add_char_to_prompt(self, base_prompt, ref_image, ref_image_2, threshold, char_threshold, exclude,
                           replacer_list, filter_toggles, char_only, use_weighting, weight_preset, custom_weight,
                           multi_char=False, det_conf=0.35):
        """Button: build CHAR blocks and merge them straight into the live prompt (no copy-paste).
        On any error/empty result, returns the prompt unchanged so we never clobber it with a warning."""
        result = self.build_char_blocks(
            ref_image, ref_image_2, threshold, char_threshold, exclude, replacer_list,
            filter_toggles, char_only, use_weighting, weight_preset, custom_weight,
            multi_char, det_conf, base_prompt=base_prompt or "",
        )
        if not isinstance(result, str) or not result.strip() or result.strip().startswith("⚠️"):
            return base_prompt  # keep the user's prompt as-is on error/empty
        return result

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
        use_weighting, weight_preset, custom_weight, replacer_list,
        filter_toggles, filter_neg_only, filter_opposites, char_only, decensor_mode,
        ref_image_2, fusion_mode,
        sort_by, sort_dir, log_batch,
        qs_enabled, nai_mode, nai_char_mode, qs_inject_mode,
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
            nai_char_mode=bool(nai_char_mode),
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
        cfg["_fusion_tags2_pos"] = None
        cfg["_fusion_tags2_char"] = None

        sig = (
            cfg["index_mode"], cfg["manual_index"],
            self._files_signature(files, cfg["pixiv_enabled"], cfg["pixiv_url"], cfg["batch_enabled"], cfg["folder_path"], cfg["sort_by"], cfg["sort_dir"])
        )
        if sig != self._counter_signature:
            self.batch_offset = 0
            self._counter_signature = sig

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
            shared.wd14_batch_image_paths = {}
            shared.wd14_batch_tag_payloads = {}
            shared.wd14_batch_neg_tags = {}

        if isinstance(cfg["ref_image"], Image.Image):
            shared.wd14_slot_image_pil = cfg["ref_image"]
            try:
                p._wd14_slot_image_pil = cfg["ref_image"]
            except Exception:
                pass
        else:
            if not hasattr(shared, "wd14_slot_image_pil"):
                shared.wd14_slot_image_pil = None

        try:
            p._wd14_batch_image_paths = shared.wd14_batch_image_paths
            p._wd14_batch_tag_payloads = shared.wd14_batch_tag_payloads
            p._wd14_batch_neg_tags = shared.wd14_batch_neg_tags
        except Exception:
            pass

        self._active_cfg = cfg
        print(f"✅ [WD14] process(): cfg stored. files={cfg['_num_files']}. Tagging will run in process_batch().")
        return

    # ======================================================
    # BATCH-SAFE TAGGING
    # ======================================================
    def process_batch(self, p, *args, **kwargs):
        cfg = self._active_cfg
        if not isinstance(cfg, dict):
            return

        prompts = kwargs.get("prompts", None)
        if prompts is None:
            for a in args:
                if isinstance(a, list) and a and all(isinstance(x, str) for x in a):
                    prompts = a
                    break
        if prompts is None:
            prompts = getattr(p, "prompts", None)

        if not (isinstance(prompts, list) and prompts and all(isinstance(x, str) for x in prompts)):
            return

        bsz = len(prompts)

        negative_prompts = kwargs.get("negative_prompts", None)
        if not (isinstance(negative_prompts, list) and len(negative_prompts) == bsz):
            negative_prompts = getattr(p, "negative_prompts", None)
        if not (isinstance(negative_prompts, list) and len(negative_prompts) == bsz):
            base_neg = getattr(p, "negative_prompt", "") or ""
            negative_prompts = [base_neg] * bsz

        base_index = self._resolve_base_index(p, prompts)
        files = cfg.get("_files", []) or []
        num_files = int(cfg.get("_num_files", 0) or 0)

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

        # QuickShot
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

        # If WD14 disabled
        if not cfg.get("enabled", False):
            try:
                # Write injected (e.g. Quick-Shot) prompts into all_prompts so they appear in
                # the saved infotext and carry over via "Send to img2img". Forge rebuilds
                # p.prompts from p.all_prompts after sampling, so setting p.prompts alone is
                # not enough — the infotext is generated from all_prompts.
                for bi in range(bsz):
                    gi = base_index + bi
                    if hasattr(p, "all_prompts") and isinstance(p.all_prompts, list) and 0 <= gi < len(p.all_prompts):
                        p.all_prompts[gi] = prompts[bi]
                    if hasattr(p, "all_negative_prompts") and isinstance(p.all_negative_prompts, list) and 0 <= gi < len(p.all_negative_prompts):
                        p.all_negative_prompts[gi] = negative_prompts[bi]
                p.prompts = prompts
                p.negative_prompts = negative_prompts
                p.prompt = prompts[0]
                p.negative_prompt = negative_prompts[0]
            except Exception:
                pass
            return

        # Load WD14 model + tags
        onnx_path = getattr(shared.opts, "wd_tagger_model_path", "")
        csv_path  = getattr(shared.opts, "wd_tagger_csv_path", "")

        if not onnx_path or not csv_path or not os.path.exists(onnx_path) or not os.path.exists(csv_path):
            print("⚠️ [WD14] Missing model_path/csv_path. Set them in Settings → WD14 Tagger.")
            return

        # ✅ model reload if path changed
        if self.model is None or self._onnx_path != onnx_path:
            try:
                self.model = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception:
                self.model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._onnx_path = onnx_path

        # ✅ FIX: tags.csv load (you were missing this entirely)
        if self.tags is None or self._csv_path != csv_path:
            try:
                self.tags = pd.read_csv(csv_path)
                self._csv_path = csv_path
                if "name" not in self.tags.columns or "category" not in self.tags.columns:
                    print("⚠️ [WD14] tags.csv missing required columns: name/category")
                    return
            except Exception as e:
                print(f"⚠️ [WD14] Failed to read tags.csv: {e}")
                return

        # Weight cap
        try:
            max_w = 1.3 if "Pony" in cfg["weight_preset"] else (2.0 if "NAI" in cfg["weight_preset"] else float(cfg["custom_weight"]))
        except Exception:
            max_w = 1.3

        # Excludes + filters
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

        # Replacements
        replacements = {}
        for line in (cfg.get("replacer_list") or "").split("\n"):
            if ":" in line:
                a, b = line.split(":", 1)
                replacements[a.strip().lower()] = b.strip().lower()

        # Fusion tags (lazy compute once)
        tags2_pos = cfg.get("_fusion_tags2_pos", None)
        tags2_char = cfg.get("_fusion_tags2_char", None)

        if tags2_pos is None or tags2_char is None:
            tags2_pos = []
            tags2_char = []

            if cfg.get("fusion_mode") != "Single Image" and isinstance(cfg.get("ref_image_2"), Image.Image):
                probs2 = self.run_inference(cfg["ref_image_2"])
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

                        if is_char:
                            tags2_char.append(tag_str)

                        if "Combine" in cfg["fusion_mode"]:
                            tags2_pos.append(tag_str)
                        elif "Replace" in cfg["fusion_mode"] and is_char:
                            tags2_pos.append(tag_str)

            cfg["_fusion_tags2_pos"] = tags2_pos
            cfg["_fusion_tags2_char"] = tags2_char

        # Main loop (this batch)
        base_counter = int(cfg.get("manual_index", 0) or 0)

        for bi in range(bsz):
            gi = base_index + bi

            img1 = None
            chosen_path = None
            idx = None

            if num_files > 0:
                if cfg["index_mode"] == "Increment":
                    idx = (base_counter + gi) % num_files
                elif cfg["index_mode"] == "Decrement":
                    idx = (base_counter - gi) % num_files
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

            shared.wd14_batch_image_paths[gi] = chosen_path if chosen_path else None

            if cfg.get("log_batch", True):
                if chosen_path:
                    print(f"[WD14] {os.path.basename(chosen_path)} ({bi+1}/{bsz}) gidx={gi} idx={idx} mode={cfg['index_mode']}")
                else:
                    print(f"[WD14] ref_image ({bi+1}/{bsz}) gidx={gi} mode={cfg['index_mode']}")

            if img1 is None:
                continue

            probs1 = self.run_inference(img1)
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

            # Fusion apply
            if tags2_pos:
                if "Combine" in cfg["fusion_mode"]:
                    final_pos = list(dict.fromkeys(final_pos + tags2_pos))
                elif "Replace" in cfg["fusion_mode"]:
                    final_pos = [t for t in final_pos if t not in img1_char_tags] + tags2_pos

            tag_payload = ", ".join(final_pos)

            shared.wd14_batch_tag_payloads[gi] = tag_payload
            shared.wd14_batch_neg_tags[gi] = ", ".join(sorted(set(final_neg))) if final_neg else ""

            use_char_blocks = bool(cfg.get("nai_char_mode"))
            char_token = _detect_char_token(prompts[bi])

            # Multi-character: when CHAR mode is on, auto-detect people in the source image and tag
            # each crop separately so each gets its own slot (CHAR1 = leftmost, CHAR2 = next, ...).
            # Reliable per-character split — no reliance on a model guessing the separation.
            multi_char_blocks = None
            if use_char_blocks:
                try:
                    _boxes, _ = self._detect_people(img1, 0.35)
                except Exception:
                    _boxes = None
                if _boxes and len(_boxes) >= 2:
                    _W, _H = img1.size
                    multi_char_blocks = []
                    for (_bx1, _by1, _bx2, _by2) in _boxes[:6]:
                        _px = int((_bx2 - _bx1) * 0.06)
                        _py = int((_by2 - _by1) * 0.06)
                        _crop = img1.crop((max(0, _bx1 - _px), max(0, _by1 - _py), min(_W, _bx2 + _px), min(_H, _by2 + _py)))
                        _cp = self.run_inference(_crop)
                        _cch = []
                        if _cp is not None:
                            for _j, _prob in enumerate(_cp):
                                _row = self.tags.iloc[_j]
                                _name = str(_row["name"]).replace("_", " ").lower()
                                _cat = int(_row["category"])
                                if _cat == 9 or _prob < (cfg["char_threshold"] if _cat == 4 else cfg["threshold"]):
                                    continue
                                if any(_ex in _name for _ex in ex_list):
                                    continue
                                if not (any(_k in _name for _k in self.char_design_keywords) or _cat == 4):
                                    continue
                                _ptag = replacements.get(_name, _name)
                                _cch.append(f"({_ptag}:{1.0 + (_prob * (max_w - 1.0)):.2f})" if cfg.get("use_weighting") else _ptag)
                        if _cch:
                            multi_char_blocks.append(_cch)
                    if multi_char_blocks and len(multi_char_blocks) >= 2:
                        print(f"[WD14] multi-char: {len(multi_char_blocks)} characters detected -> CHAR1..CHAR{len(multi_char_blocks)}")
                    else:
                        multi_char_blocks = None

            if use_char_blocks:
                if multi_char_blocks:
                    # One CHAR slot per detected person (reliable per-character split, left-to-right).
                    char_blocks = multi_char_blocks
                    replace_slots = set()
                else:
                    tags2_char = cfg.get("_fusion_tags2_char", []) or []

                    if "Replace" in cfg["fusion_mode"] and tags2_char:
                        char_blocks = [tags2_char]
                        replace_slots = {0}
                    elif "Combine" in cfg["fusion_mode"] and tags2_char:
                        char_blocks = [img1_char_tags, tags2_char]
                        replace_slots = set()
                    else:
                        char_blocks = [img1_char_tags]
                        replace_slots = set()

                char_set = set()
                for c in char_blocks:
                    for t in (c or []):
                        char_set.add(t)

                global_tags = [t for t in final_pos if t not in char_set]
                global_payload = ", ".join(global_tags)

                prompts[bi] = inject_global_before_chars(
                    prompts[bi],
                    global_payload,
                    mode=cfg["mode"],
                    token=char_token
                )

                prompts[bi] = merge_chars_into_prompt(
                    prompts[bi],
                    [c for c in char_blocks if c],
                    token=char_token,
                    max_chars=6,
                    replace_indices=(sorted(replace_slots) if replace_slots else None),
                )
            else:
                prompts[bi] = inject_global_before_chars(
                    prompts[bi],
                    tag_payload,
                    mode=cfg["mode"],
                    token=char_token
                )

            if final_neg:
                add = ", ".join(sorted(set(final_neg)))
                base = (negative_prompts[bi] or "").strip(" ,")
                negative_prompts[bi] = (base + ", " + add).strip(" ,")

            try:
                if hasattr(p, "all_prompts") and isinstance(p.all_prompts, list) and 0 <= gi < len(p.all_prompts):
                    p.all_prompts[gi] = prompts[bi]
                if hasattr(p, "all_negative_prompts") and isinstance(p.all_negative_prompts, list) and 0 <= gi < len(p.all_negative_prompts):
                    p.all_negative_prompts[gi] = negative_prompts[bi]
            except Exception:
                pass

        try:
            # Final sync: make sure every image's final prompt is in all_prompts (covers images
            # the loop above skipped via 'continue', e.g. no source image but Quick-Shot tags
            # applied), so the saved infotext / "Send to img2img" reflects the injected prompt.
            for bi in range(bsz):
                gi = base_index + bi
                if hasattr(p, "all_prompts") and isinstance(p.all_prompts, list) and 0 <= gi < len(p.all_prompts):
                    p.all_prompts[gi] = prompts[bi]
                if hasattr(p, "all_negative_prompts") and isinstance(p.all_negative_prompts, list) and 0 <= gi < len(p.all_negative_prompts):
                    p.all_negative_prompts[gi] = negative_prompts[bi]
            p.prompts = prompts
            p.negative_prompts = negative_prompts
            p.prompt = prompts[0]
            p.negative_prompt = negative_prompts[0]
        except Exception:
            pass

        print(f"✅ [WD14] process_batch(): injected {bsz} prompts (base_index={base_index}).")
        return
