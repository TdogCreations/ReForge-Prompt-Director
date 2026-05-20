# wd_quickshot_helper.py
import random
from typing import Optional, Callable, Tuple

# ==========================================================
# AGE MAP
# ==========================================================
AGE_PROMPT_MAP = {
    # --- minors (allowed for non-explicit content; explicit rating is blocked) ---
    "Newborn": "newborn, baby, infant, tiny, very young",
    "Baby": "baby, infant, very young",
    "Young Child": "toddler, young child, child, small body, short legs, short arms",
    "Child": "child, young, small body, childlike proportions",
    "Pre-Teen": "pre-teen, young teen, childlike, early puberty",
    "Teen": "teen, youthful, young-looking",

    # --- adults ---
    "Young Adult": "young adult, 18+, youthful adult, adult proportions",
    "Adult": "adult, 23+, adult proportions, mature body",
    "Older Adult": "older adult, 50+, mature, wrinkles, aging features",
}

ADULT_AGE_GROUPS = {"Young Adult", "Adult", "Older Adult"}

TIME_OF_DAY_MAP = {
    "dawn": "dawn, early morning, soft light, golden hour, sunrise, warm glow, gentle lighting, natural light, soft shadows",
    "midday": "midday, bright daylight, harsh lighting, sharp shadows, vivid colors, high contrast, intense light",
    "sunset": "sunset, golden hour, warm lighting, orange glow, dusk, cinematic lighting, dramatic shadows, ambient lighting",
    "midnight": "midnight, night, low light, moonlight, dark ambiance, noir lighting, shadowy, mysterious atmosphere",
}

VIEWPOINT_MAP = {
    "from above": "high angle, from above",
    "from below": "low angle, from below",
    "from side": "side view, profile view",
    "from behind": "rear view, from behind",
    "from the front": "front view, from the front",
}


def _age_strength_to_weight(age_strength: int, nai_mode: bool) -> Optional[float]:
    try:
        s = int(age_strength)
    except Exception:
        s = 0

    s = max(0, min(10, s))
    if s <= 0:
        return None

    if nai_mode:
        w = 1.0 + (s * 0.5)   # 10 -> 6.0
        return min(6.0, w)

    w = 1.0 + (s * 0.07)      # 10 -> 1.7
    return min(1.7, w)


def _split_tags(phrase: str) -> list[str]:
    if not phrase:
        return []
    return [t.strip() for t in phrase.split(",") if t.strip()]


def apply_quickshot(
    prompt: str,
    negative_prompt: str,
    *,
    inject_mode: str,
    nai_mode: bool,

    rating_val: int,
    io_val: int,
    view_val: int,

    time_of_day: str,
    time_weight: float,

    vol_light: str,
    vol_weight: float,

    viewpoint: str,
    viewpoint_scale: int,
    dutch_angle: bool,

    fg_blur: int,
    bg_blur: int,

    age_group: str,
    age_strength: int,

    remove_baby_props: bool,

    maybe_weight: Callable[[str, float, bool], str],
    clamp_weight_xl: Callable[[float], float],
) -> Tuple[str, str]:

    prompt = prompt or ""
    negative_prompt = negative_prompt or ""

    auto_tags: list[str] = []
    auto_neg: list[str] = []
    mult = 0.40 if nai_mode else 0.10

    # Guardrail: if minor + explicit requested, block explicit
    is_minor = bool(age_group and age_group != "None" and age_group not in ADULT_AGE_GROUPS)
    if rating_val > 0:
        rating_val = 0
        if "rating:explicit" not in (negative_prompt or ""):
            negative_prompt = (negative_prompt + ", rating:explicit").strip(", ")

    # Suppress baby props
    if remove_baby_props:
        if nai_mode:
            auto_tags.append("(pacifier:-3)")
            auto_tags.append("(bib:-3)")
        else:
            auto_neg.append("(pacifier:1.5)")
            auto_neg.append("(bib:1.5)")

    # Age group + strength
    if age_group and age_group != "None":
        phrase = AGE_PROMPT_MAP.get(age_group)
        w_age = _age_strength_to_weight(age_strength, nai_mode)
        if phrase and w_age is not None:
            for t in _split_tags(phrase):
                auto_tags.append(f"({t}:{float(w_age):.2f})")

    # Rating
    if rating_val > 0:
        w = 1.0 + (float(rating_val) * mult)
        if not nai_mode:
            w = clamp_weight_xl(w)
        auto_tags.append(maybe_weight("rating:explicit", w, nai_mode=nai_mode))
    elif rating_val < 0:
        if nai_mode:
            w = 1.0 + (abs(float(rating_val)) * mult * 2)
            auto_neg.append(maybe_weight("rating:explicit", w, nai_mode=True))
        else:
            negative_prompt = (negative_prompt + ", rating:explicit").strip(", ")

    # Indoor/Outdoor
    if io_val > 0:
        w = 1.0 + (float(io_val) * mult)
        if not nai_mode:
            w = clamp_weight_xl(w)
        auto_tags.append(maybe_weight("indoor", w, nai_mode=nai_mode))
    elif io_val < 0:
        w = 1.0 + (abs(float(io_val)) * mult)
        if not nai_mode:
            w = clamp_weight_xl(w)
        auto_tags.append(maybe_weight("outdoor", w, nai_mode=nai_mode))

    # Multiple views
    if view_val > 0:
        w = 1.0 + (float(view_val) * mult)
        if not nai_mode:
            w = clamp_weight_xl(w)
        auto_tags.append(maybe_weight("multiple views", w, nai_mode=nai_mode))
        auto_tags.append("split view")

    # Time of day
    t_choice = random.choice(["dawn", "midday", "sunset", "midnight"]) if time_of_day == "Random" else time_of_day
    if t_choice and t_choice != "None" and float(time_weight) > 0:
        phrase = TIME_OF_DAY_MAP.get(t_choice, t_choice)
        t_tags = _split_tags(phrase)
        if nai_mode:
            for t in t_tags:
                auto_tags.append(maybe_weight(t, float(time_weight), nai_mode=True))
        else:
            w = clamp_weight_xl(1.0 + (float(time_weight) * 0.1))
            for t in t_tags:
                auto_tags.append(maybe_weight(t, w, nai_mode=False))

    # Lighting
    l_choice = random.choice([
        "from the left", "from the right", "from above", "from the side", "from below", "from behind"
    ]) if vol_light == "Random" else vol_light

    if l_choice and l_choice != "None" and float(vol_weight) > 0:
        phrase = f"volumetric lighting {l_choice}"
        if nai_mode:
            auto_tags.append(maybe_weight(phrase, float(vol_weight), nai_mode=True))
        else:
            w = clamp_weight_xl(1.0 + (float(vol_weight) * 0.1))
            auto_tags.append(maybe_weight(phrase, w, nai_mode=False))

    # Viewpoint + scale
    if viewpoint == "Random":
        vp_choice = random.choice(["from above", "from below", "from side", "from behind", "from the front"])
    elif viewpoint and viewpoint != "None":
        vp_choice = viewpoint
    else:
        vp_choice = None

    if vp_choice:
        phrase = VIEWPOINT_MAP.get(vp_choice, vp_choice)
        vp_tags = _split_tags(phrase)
        try:
            vs = float(viewpoint_scale)
        except Exception:
            vs = 0.0

        if nai_mode:
            if vs > 0:
                for t in vp_tags:
                    auto_tags.append(maybe_weight(t, vs, nai_mode=True))
            else:
                auto_tags.extend(vp_tags)
        else:
            if vs > 0:
                w = clamp_weight_xl(1.0 + 0.1 * vs)
                for t in vp_tags:
                    auto_tags.append(maybe_weight(t, w, nai_mode=False))
            else:
                auto_tags.extend(vp_tags)

    if dutch_angle:
        auto_tags.append("dutch angle")

    # Blur
    if fg_blur > 0:
        w = 1.0 + (float(fg_blur) * mult)
        if not nai_mode:
            w = clamp_weight_xl(w)
        auto_tags.append(maybe_weight("blurry foreground", w, nai_mode=nai_mode))
        auto_tags.append("foreground bokeh")
    elif fg_blur < 0:
        if nai_mode:
            w = 1.0 + (abs(float(fg_blur)) * mult)
            auto_neg.append(maybe_weight("blurry foreground", w, nai_mode=True))
            auto_neg.append(maybe_weight("foreground bokeh", w, nai_mode=True))
        else:
            auto_tags.append("sharp focus")
            auto_tags.append("sharp foreground")

    if bg_blur > 0:
        w = 1.0 + (float(bg_blur) * mult)
        if not nai_mode:
            w = clamp_weight_xl(w)
        auto_tags.append(maybe_weight("depth of field", w, nai_mode=nai_mode))
        auto_tags.append(maybe_weight("blurry background", w, nai_mode=nai_mode))
    elif bg_blur < 0:
        if nai_mode:
            w = 1.0 + (abs(float(bg_blur)) * mult)
            auto_neg.append(maybe_weight("depth of field", w, nai_mode=True))
            auto_neg.append(maybe_weight("blurry background", w, nai_mode=True))
        else:
            auto_tags.append("deep focus")
            auto_tags.append("sharp background")

    # Inject
    pos_payload = ", ".join([t for t in auto_tags if t]).strip()
    neg_payload = ", ".join([t for t in auto_neg if t]).strip()

    if pos_payload:
        if inject_mode == "Prepend":
            prompt = f"{pos_payload}, {prompt}".strip(", ")
        elif inject_mode == "Replace":
            prompt = pos_payload
        else:
            prompt = f"{prompt}, {pos_payload}".strip(", ")

    if neg_payload:
        negative_prompt = f"{negative_prompt}, {neg_payload}".strip(", ")

    return prompt, negative_prompt
