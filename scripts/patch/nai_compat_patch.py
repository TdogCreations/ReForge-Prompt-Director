# nai_compat_patch.py
# ReForge / A1111 NAI compat: ensure NAI reads the *post-injection* prompts (p.all_prompts)
# Works by wrapping NAI processing entrypoints that receive (p, prompts, ...)

import importlib
import inspect
import traceback

_PATCHED = False


def _resolve_base_index(p, prompts):
    """Find where this batch's prompts live inside p.all_prompts (handles ReForge recreating p)."""
    try:
        allp = getattr(p, "all_prompts", None)
        if isinstance(allp, list) and isinstance(prompts, (list, tuple)) and len(prompts) > 0:
            bsz = len(prompts)
            limit = len(allp) - bsz
            for i in range(0, max(-1, limit) + 1):
                if allp[i:i + bsz] == list(prompts):
                    return i
    except Exception:
        pass

    # fallback
    try:
        bi = getattr(p, "batch_index", None)
        if isinstance(bi, int):
            return bi * len(prompts)
    except Exception:
        pass

    return 0


def _maybe_sync_prompts_from_all_prompts(p, prompts):
    """If p.all_prompts exists, overwrite prompts[] with the authoritative values from p.all_prompts."""
    if not isinstance(prompts, list):
        return False

    allp = getattr(p, "all_prompts", None)
    if not isinstance(allp, list) or not allp:
        return False

    bsz = len(prompts)
    if bsz <= 0:
        return False

    base = _resolve_base_index(p, prompts)
    changed = False

    for i in range(bsz):
        gi = base + i
        if 0 <= gi < len(allp):
            ap = allp[gi]
            if isinstance(ap, str) and ap and ap != prompts[i]:
                prompts[i] = ap
                changed = True

    if changed:
        try:
            print(f"✅ [NAI PATCH] Synced prompts from p.all_prompts (base_index={base}, batch={bsz})")
        except Exception:
            pass

    return changed


def _wrap_entrypoint(mod, fn_name):
    original = getattr(mod, fn_name, None)
    if not callable(original):
        return False

    # avoid wrapping twice
    if getattr(original, "_nai_patch_wrapped", False):
        return False

    sig = None
    try:
        sig = inspect.signature(original)
    except Exception:
        sig = None

    def wrapper(*args, **kwargs):
        # Try to locate (p, prompts) in args/kwargs
        p = None
        prompts = None

        # kwargs first
        if "p" in kwargs:
            p = kwargs.get("p")
        if "prompts" in kwargs:
            prompts = kwargs.get("prompts")

        # args fallback
        if p is None and len(args) >= 1:
            p = args[0]

        if prompts is None:
            # common patterns: (p, prompts, ...) or prompts passed as any list[str]
            if len(args) >= 2 and isinstance(args[1], list) and (not args[1] or isinstance(args[1][0], str)):
                prompts = args[1]
            else:
                for a in args:
                    if isinstance(a, list) and (not a or isinstance(a[0], str)):
                        prompts = a
                        break

        # sync
        try:
            if p is not None and prompts is not None:
                _maybe_sync_prompts_from_all_prompts(p, prompts)
        except Exception:
            print("⚠️ [NAI PATCH] Sync attempt failed:")
            traceback.print_exc()

        return original(*args, **kwargs)

    wrapper._nai_patch_wrapped = True
    setattr(mod, fn_name, wrapper)
    return True


def apply_nai_patch():
    """
    Patch NAI processing module entrypoints so NAI uses prompts after JoyCaption/WD14 injection.
    Safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return True

    # Import the NAI processing module (your log shows this exact one)
    try:
        mod = importlib.import_module("nai_api_gen.nai_api_processing")
    except Exception as e:
        print(f"⚠️ [NAI PATCH] Could not import nai_api_gen.nai_api_processing: {e}")
        return False

    # Entry points to try (different forks use different names)
    candidates = [
        "process",
        "process_batch",
        "process_images",
        "run",
        "run_batch",
        "txt2img",
        "img2img",
        "nai_process",
        "nai_process_batch",
        "generate",
        "generate_batch",
        "create_payload",
        "build_payload",
        "make_payload",
        "request",
    ]

    wrapped_any = False
    for name in candidates:
        try:
            if _wrap_entrypoint(mod, name):
                print(f"✅ [NAI PATCH] Wrapped entrypoint: {name}")
                wrapped_any = True
        except Exception:
            pass

    if wrapped_any:
        _PATCHED = True
        print("✅ [NAI PATCH] Patched NAI processing module: nai_api_gen.nai_api_processing")
        return True

    # If we found module but nothing to wrap, still report clearly
    print("⚠️ [NAI PATCH] Loaded nai_api_processing, but found no known entrypoints to wrap.")
    return False
